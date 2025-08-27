# coding=utf-8
# Copyright (c) 2025, HUAWEI CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import datetime
import pickle
import torch
import torch.distributed as dist
import vllm.distributed.parallel_state as ps
from vllm import RequestOutput, CompletionOutput
from vllm.sequence import Logprob
from vllm.entrypoints.llm import LLM, LLMEngine
from .config import RolloutRebalanceConfig
from .utils import CheckCounter, hook, rank_log_info


class RolloutRebalanceEngine(object):
    device = 'npu'

    def __init__(self, llm_engine=None, check_interval=200):
        self.llm_engine = llm_engine
        self.rank = dist.get_rank()
        self.rebalance_counter = CheckCounter(check_interval)
        self.dp_group = self.llm_engine.dp_group
        self.world_size = len(ps.get_dp_group().ranks)

        # 为了让Seq能独立搬迁，此处先移除request的父子关系，最后将在self.recover中还原父子关系
        for req_state in self.llm_engine.output_processor.request_states.values():
            req_state.parent_req = None

    @staticmethod
    def _get_bs(size):
        for bs in States.graph_batch_sizes[::-1]:
            if size <= bs:
                return bs
        return size

    @staticmethod
    def _split_rebalance_outputs(outputs):
        rebalance_outputs = []
        current_outputs = []
        for request_output in outputs:
            if not request_output.request_id.startswith('src_rank'):
                current_outputs.append(request_output)
                continue
            src_rank, src_request_id = request_output.request_id.split('src_rank_', 1)[-1].split('_', 1)
            seq_output = request_output.outputs[0]
            logprobs_cache = dict(
                logprobs=[],
                cumulative_logprob=seq_output.cumulative_logprob,
            )
            for item in seq_output.logprobs:
                key = list(item.keys())[0]
                value = list(item.values())[0]
                logprobs_cache['logprobs'].append([key, value.logprob, value.rank])

            rebalance_outputs.append(dict(
                rank=int(src_rank),
                req_id=src_request_id,
                prompt_token_ids=request_output.prompt_token_ids,
                token_ids=seq_output.token_ids,
                **logprobs_cache,
            ))
        return rebalance_outputs, current_outputs

    @staticmethod
    def _build_rebalance_request_output(rebalance_output):
        return RequestOutput(
            request_id=rebalance_output['req_id'],
            prompt=None,
            prompt_token_ids=rebalance_output['prompt_token_ids'],
            prompt_logprobs=None,
            outputs=[
                CompletionOutput(
                    index=rebalance_output['req_id'].split('_', 1)[0],
                    text='',
                    token_ids=rebalance_output['token_ids'],
                    cumulative_logprob=rebalance_output['cumulative_logprob'],
                    logprobs=[{token_id: Logprob(logprob=logprob, rank=rank, decoded_token=None)
                               for token_id, logprob, rank in rebalance_output['logprobs']}],
                    finish_reason='stop'
                )
            ],
            finished=True,
        )

    def calc_balancing_tasks(self, remaining_reqs_by_rank):
        """
        计算最优的请求迁移任务清单，以实现负载均衡。

        最优策略定义为三层目标：
        1. 主要目标：将整个DP组所需的最大档位(max_bs)降至最低。
        2. 次要目标：在满足1的前提下，使迁移的请求数量(cost)最少。
        3. 补充目标：在满足1和2的前提下，使得各个rank间的数据搬迁流向尽可能均匀，避免多对一阻塞。
        """
        if not remaining_reqs_by_rank or len(remaining_reqs_by_rank) <= 1:
            return []

        rank_req_ids_map = {item['rank']: list(item['req_ids']) for item in remaining_reqs_by_rank}
        rank_req_cnt_map = {rank: len(ids) for rank, ids in rank_req_ids_map.items()}
        ranks = sorted(rank_req_cnt_map.keys())

        # 寻找所有可行的优化方案
        max_bs_before = max(self._get_bs(bs) for bs in rank_req_cnt_map.values())
        avg_bs = sum(rank_req_cnt_map.values()) / len(rank_req_cnt_map.values())
        max_bs_next = None
        for target_bs in States.graph_batch_sizes[::-1]:  # 优先匹配出最小的BS
            if avg_bs <= target_bs < max_bs_before:
                max_bs_next = target_bs
                break

        if not max_bs_next:
            return []

        # 分析各个worker的可调度量
        donors = []
        receivers = []
        for rank in ranks:
            req_cnt = rank_req_cnt_map[rank]
            delta = req_cnt - max_bs_next
            if delta > 0:
                donors.append(dict(
                    rank=rank,
                    surplus=delta,
                    req_ids=rank_req_ids_map[rank][:delta],
                ))
            elif delta < 0:
                receivers.append(dict(
                    rank=rank,
                    capacity=-delta,
                ))

        # 生成迁移任务清单：为了让迁移速度更快，尽可能将请求均衡分发给目标worker，但优先分发给最空闲的worker（即容量最大的worker）
        balancing_tasks = []
        while True:
            donor_index = 0
            for receiver in sorted(receivers, key=lambda r: r['capacity'], reverse=True):
                donor = donors[donor_index]
                num_to_move = min(donor['surplus'], receiver['capacity'])
                balancing_tasks += [dict(
                    from_rank=donor['rank'],
                    to_rank=receiver['rank'],
                    req_id=req_id
                ) for req_id in donor['req_ids'][:num_to_move]]

                donor['req_ids'] = donor['req_ids'][num_to_move:]
                donor['surplus'] -= num_to_move
                receiver['capacity'] -= num_to_move
                donor_index = (donor_index + 1) % len(donors)

            donors = [x for x in donors if x['surplus']]
            if not donors:
                break
            receivers = [x for x in receivers if x['capacity']]

        return balancing_tasks

    def check(self):
        need_profile = self.ProfileCache.enable and self.ProfileCache.counter.check()
        need_rebalance_check = self.rebalance_counter.check()
        if not (need_profile or need_rebalance_check):
            return
        start = datetime.datetime.now(tz=datetime.timezone.utc)
        group_states = self.sync_group_states()
        if need_profile:
            self.profile(group_states)
        if not need_rebalance_check:  # 控制Rebalance检测的平度
            return
        schedule_tasks = self.calc_balancing_tasks(group_states)
        ranks_info = [f'[Rank={x["rank"]}, Remain={len(x["req_ids"])}]' for x in group_states]
        if schedule_tasks:
            rank_log_info(f'[RebalanceScheduleTasks][Cnt={len(schedule_tasks)}]')
            for schedule_task in schedule_tasks:
                rank_log_info(
                    f'[Rebalance][ReqId={schedule_task["req_id"]}]'
                    f'[Src={schedule_task["from_rank"]}][Dst={schedule_task["to_rank"]}]')
            self.all_to_all_v_tasks(schedule_tasks)
        cost = round((datetime.datetime.now(tz=datetime.timezone.utc) - start).total_seconds() * 1000, 1)
        rank_log_info(f'[RebalanceSchedule][Cost={cost}ms] {"".join(ranks_info)}')

    def get_rebalance_outputs(self, rebalance_outputs):
        """
        通过all_gather同步，获取属于当前rank的rebalance_output
        """
        ret = []
        if self.world_size <= 1:
            return ret
        full_rebalance_outputs = [[] for _ in range(self.world_size)]
        dist.all_gather_object(full_rebalance_outputs, rebalance_outputs, group=self.dp_group)
        for rank_rebalance_outputs in full_rebalance_outputs:
            for rebalance_output in rank_rebalance_outputs:
                if self.rank != rebalance_output['rank']:
                    continue
                ret.append(rebalance_output)
        return ret

    def recover(self, outputs):
        # 将进行了rebalance迁移的outputs通过all_gather还原到源rank
        rebalance_outputs, current_outputs = self._split_rebalance_outputs(outputs)
        for rebalance_output in self.get_rebalance_outputs(rebalance_outputs):
            rank_log_info(f'[RecvReqId={rebalance_output["req_id"]}]', force=True)
            current_outputs.append(self._build_rebalance_request_output(rebalance_output))

        current_outputs_map = {}

        for request_output in current_outputs:
            request_id = request_output.request_id
            if '_' in request_id:
                request_id = request_id.split('_', 1)[-1]

            parent_request = current_outputs_map.get(request_id)
            if not parent_request:
                parent_request = RequestOutput(
                    request_id=request_id,
                    prompt=None,
                    prompt_token_ids=request_output.prompt_token_ids,
                    prompt_logprobs=None,
                    outputs=[],
                    finished=True,
                )
                current_outputs_map[request_id] = parent_request
            parent_request.outputs += request_output.outputs

        return current_outputs_map.values()

    def get_current_state(self):
        return dict(
            rank=self.rank,
            req_ids=list(self.llm_engine.output_processor.request_states.keys()),
        )

    def sync_group_states(self):
        if self.world_size == 1:
            return [self.get_current_state()]
        group_states = [None for _ in range(self.world_size)]
        dist.all_gather_object(group_states, self.get_current_state(), group=self.dp_group)
        return group_states

    def all_to_all_v_tasks(self, schedule_tasks):
        objects_to_send = [[] for _ in range(dist.get_world_size())]
        send_tasks = []
        for schedule_task in schedule_tasks:
            if self.rank == schedule_task['from_rank']:
                request_task = RebalanceRequestTask(self.llm_engine).load_by_req_id(schedule_task['req_id'])
                send_tasks.append((request_task, schedule_task['to_rank']))
                objects_to_send[schedule_task['to_rank']].append(request_task.get_transfer_dict())
                request_task.trigger_abort()

        tensor_list = [
            torch.frombuffer(pickle.dumps(data), dtype=torch.uint8).to(self.device)
            for data in objects_to_send]
        local_sizes = torch.tensor([len(t) for t in tensor_list], dtype=torch.long).to(self.device)
        remote_sizes = torch.empty_like(local_sizes).to(self.device)
        dist.all_to_all_single(remote_sizes, local_sizes)
        input_tensor = torch.cat(tensor_list).to(self.device)
        output_tensor = torch.empty(remote_sizes.sum().item(), dtype=torch.uint8).to(self.device)
        dist.all_to_all_single(
            output_tensor,
            input_tensor,
            output_split_sizes=remote_sizes.tolist(),
            input_split_sizes=local_sizes.tolist(),
        )
        received_tensor = torch.split(output_tensor, remote_sizes.tolist())
        received_tasks = []
        for rank_data in received_tensor:
            received_tasks += pickle.loads(rank_data.to('cpu').numpy().tobytes())
        self.send_kv_caches(send_tasks)
        self.load_received_tasks(received_tasks)

    def send_kv_caches(self, send_tasks):
        if States.reprefill_mode:
            return
        for request_task, to_rank in send_tasks:
            for kv_cache_block in request_task.layers_kv_cache_blocks:
                dist.send(kv_cache_block, dst=to_rank)
            rank_log_info(
                f'[TaskSendKvCache][ToRank={to_rank}][ReqId={request_task.req_id}]'
            )


    def load_received_tasks(self, received_tasks):
        for request_task_dict in received_tasks:
            send_time = datetime.datetime.strptime(
                request_task_dict['send_time'],
                '%Y-%m-%d %H:%M:%S.%f').replace(tzinfo=datetime.timezone.utc)
            task_recv_time = datetime.datetime.now(tz=datetime.timezone.utc)
            cost = round((task_recv_time - send_time).total_seconds() * 1000, 1)
            from_rank = request_task_dict["src_rank"]
            rank_log_info(
                f'[ReceivedTask][FromRank={from_rank}]'
                f'[Cost={cost}ms][SendTime={request_task_dict["send_time"]}]',
                force=True)

            layers_kv_cache_blocks = []

            if not States.reprefill_mode:
                for kv_cache_block_shape in request_task_dict['layers_kv_cache_shapes']:
                    block = torch.empty(kv_cache_block_shape, dtype=torch.bfloat16).cuda()
                    dist.recv(block, src=from_rank)
                    layers_kv_cache_blocks.append(block)

                cost = round(
                    (datetime.datetime.now(tz=datetime.timezone.utc) - task_recv_time).total_seconds() * 1000, 1)
                rank_log_info(
                    f'[ReceivedKvCache][FromRank={from_rank}][Cost={cost}ms]'
                    f'[KV_CACHE_SHAPE_LIST={request_task_dict["layers_kv_cache_shapes"]}]',
                    force=True)

            start_time = datetime.datetime.now(tz=datetime.timezone.utc)
            request_task = RebalanceRequestTask(self.llm_engine).load_by_transfer_info(
                request_task_dict=request_task_dict,
                layers_kv_cache_blocks=layers_kv_cache_blocks)

            request_task.trigger_load()
            cost = round(
                (datetime.datetime.now(tz=datetime.timezone.utc) - start_time).total_seconds() * 1000, 1)
            rank_log_info(
                f'[ReceivedTaskLoaded][FromRank={from_rank}][Cost={cost}ms]',
                force=True)

    class ProfileCache:
        enable = False
        last_time = None
        start_time = None
        max_bs = None
        times = 0
        counter = CheckCounter(100)

        @classmethod
        def start(cls, profile_interval=None):
            cls.enable = True
            now = datetime.datetime.now(tz=datetime.timezone.utc)
            cls.last_time = now
            cls.start_time = now
            cls.max_bs = None
            cls.times = 0
            if profile_interval:
                cls.counter.threshold = profile_interval
            cls.counter.reset()

    def profile(self, group_states):
        self.ProfileCache.times += 1
        now = datetime.datetime.now(tz=datetime.timezone.utc)
        cost = (now - self.ProfileCache.last_time).total_seconds()
        duration = (now - self.ProfileCache.start_time).total_seconds()
        self.ProfileCache.last_time = now
        bs_list = []
        seq_cnt_list = []
        for state_info in group_states:
            seq_cnt = len(state_info['req_ids'])
            bs_list.append(self._get_bs(seq_cnt))
            seq_cnt_list.append(seq_cnt)
        max_bs = max(bs_list)
        msgs = [
            f'[ProfileIndex={self.ProfileCache.times}]',
            f'[TotalSeqs={sum(seq_cnt_list)}]',
            f'[CurrentMaxBS={max_bs}]',
            f'[ProfileStepCost={round(cost, 1)}s]',
            f'[ProfileDuration={round(duration, 1)}s]',
            f'[TPOT={round(cost * 1000 / self.ProfileCache.counter.threshold, 1)}ms]',
        ]
        if max_bs != self.ProfileCache.max_bs:
            msgs.append(f'[MaxBSChanged: {self.ProfileCache.max_bs} -> {max_bs}]')
            self.ProfileCache.max_bs = max_bs

        rank_log_info(f'[BSMap]: {bs_list}')
        rank_log_info(f'[SeqCntMap]: {seq_cnt_list}')
        rank_log_info(''.join(msgs))


class RebalanceRequestTask:
    req_id: str
    prompt_token_ids: list
    output_token_ids: list
    logprobs_processor_cache: dict
    layers_kv_cache_blocks: list
    src_rank: int = None
    max_tokens: int

    def __init__(self, llm_engine: LLMEngine):
        self.llm_engine = llm_engine
        self.output_processor = self.llm_engine.output_processor
        self.engine_core = self.llm_engine.engine_core.engine_core
        self.scheduler = self.engine_core.scheduler
        self.model_runner = self.engine_core.model_executor.driver_worker.worker.model_runner
        self.global_kv_caches = self.model_runner.kv_caches

    def load_by_req_id(self, req_id):
        self.req_id = req_id
        input_batch = self.model_runner.input_batch
        block_table = input_batch.block_table[0].block_table
        batch_index = input_batch.req_id_to_index[req_id]
        request_block_table = torch.tensor([x for x in block_table[batch_index].tolist() if x])
        request_state = self.output_processor.request_states[req_id]
        engine_core_request = self.scheduler.requests[req_id]

        self.max_tokens = engine_core_request.max_tokens

        self.prompt_token_ids = request_state.prompt_token_ids
        self.output_token_ids = engine_core_request.output_token_ids

        # request级的kvCache采集
        self.layers_kv_cache_blocks = []
        for cache_block_index in range(len(self.global_kv_caches[0])):
            self.layers_kv_cache_blocks.append(
                torch.stack([layer[cache_block_index][request_block_table] for layer in self.global_kv_caches])
            )

        self.logprobs_processor_cache = dict(
            logprobs=[],
            cumulative_logprob=request_state.logprobs_processor.cumulative_logprob,
        )

        for item in request_state.logprobs_processor.logprobs:
            key = list(item.keys())[0]
            value = list(item.values())[0]
            self.logprobs_processor_cache['logprobs'].append([key, value.logprob, value.rank])

        return self

    def get_transfer_dict(self):
        return dict(
            req_id=self.req_id,
            src_rank=States.rebalance_engine.rank,
            prompt_token_ids=self.prompt_token_ids,
            output_token_ids=self.output_token_ids,
            max_tokens=self.max_tokens,
            logprobs_processor_cache=self.logprobs_processor_cache,
            layers_kv_cache_shapes=[list(item.shape) for item in self.layers_kv_cache_blocks],
            send_time=datetime.datetime.now(tz=datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f"),
        )

    def load_by_transfer_info(self,
                              request_task_dict: dict,
                              layers_kv_cache_blocks: list):
        self.src_rank = request_task_dict['src_rank']
        self.req_id = request_task_dict['req_id']
        self.max_tokens = request_task_dict['max_tokens']
        self.prompt_token_ids = request_task_dict['prompt_token_ids']
        self.output_token_ids = request_task_dict['output_token_ids']
        self.logprobs_processor_cache = request_task_dict['logprobs_processor_cache']
        self.layers_kv_cache_blocks = layers_kv_cache_blocks
        return self

    def trigger_abort(self):
        """
        用于在当前rank的worker内中止当前请求的执行
        """
        request_state = self.output_processor.request_states[self.req_id]
        self.llm_engine.abort_request([self.req_id])
        self.llm_engine.engine_core.abort_requests([self.req_id])
        return request_state

    def recover_request_state(self, new_req_id):
        request_state = self.output_processor.request_states[new_req_id]
        request_state.is_prefilling = False
        request_state.logprobs_processor.cumulative_logprob = \
            self.logprobs_processor_cache['cumulative_logprob']
        request_state.detokenizer.token_ids.extend(self.output_token_ids)
        request_state.logprobs_processor.logprobs = [
            {token_id: Logprob(logprob=logprob, rank=rank, decoded_token=None)}
            for token_id, logprob, rank in self.logprobs_processor_cache['logprobs']
        ]

    def recover_scheduler_request(self, new_req_id, sampling_params):
        scheduler_request = self.scheduler.requests[new_req_id]
        scheduler_request.max_tokens = self.max_tokens
        scheduler_request.append_output_token_ids(self.output_token_ids)
        scheduler_request.sampling_params = sampling_params
        from vllm.v1.request import RequestStatus
        if States.reprefill_mode:
            scheduler_request.num_computed_tokens = 0
            scheduler_request.status = RequestStatus.PREEMPTED
        else:
            scheduler_request.num_computed_tokens = scheduler_request.num_tokens - 1
            scheduler_request.status = RequestStatus.RUNNING
            self.scheduler.waiting.remove(scheduler_request)
            self.scheduler.running.append(scheduler_request)
        return scheduler_request

    def recover_model_runner(self, scheduler_request):
        if States.reprefill_mode:
            new_block_ids = []
        else:
            new_blocks = self.scheduler.kv_cache_manager.allocate_slots(scheduler_request, 1)
            new_block_ids = new_blocks.get_block_ids()
        sampling_params = scheduler_request.sampling_params
        from vllm.sampling_params import SamplingType
        if sampling_params.sampling_type == SamplingType.RANDOM_SEED:
            generator = torch.Generator(device=self.model_runner.device)
            generator.manual_seed(sampling_params.seed)
        else:
            generator = None

        from vllm.v1.worker.gpu_input_batch import CachedRequestState
        model_runner_req_state = CachedRequestState(
            req_id=scheduler_request.request_id,
            prompt_token_ids=scheduler_request.prompt_token_ids,
            mm_inputs=scheduler_request.mm_inputs,
            mm_positions=scheduler_request.mm_positions,
            sampling_params=scheduler_request.sampling_params,
            generator=generator,
            block_ids=new_block_ids,
            num_computed_tokens=scheduler_request.num_computed_tokens,
            output_token_ids=scheduler_request.output_token_ids,
            lora_request=scheduler_request.lora_request,
        )
        self.model_runner.requests[scheduler_request.request_id] = model_runner_req_state

        if not States.reprefill_mode:
            self.model_runner.input_batch.add_request(model_runner_req_state, )
            self.model_runner.input_batch.refresh_sampling_metadata()
            # kvCache还原
            for layer_index, layer_caches in enumerate(self.model_runner.kv_caches):
                reload_indexes = list(range(len(new_block_ids)))
                for i, cache_block in enumerate(self.layers_kv_cache_blocks):
                    layer_caches[i][new_block_ids] = cache_block[layer_index][reload_indexes]

    def trigger_load(self):
        """
        将任务添加到当前rank worker所在的LLMEngine中，并恢复调度
        """
        # 复制继承自其他请求的采样策略
        import copy
        sampling_params = copy.copy(States.sampling_params)
        new_req_id = self.req_id if self.req_id.startswith('src_rank') \
            else f'src_rank_{self.src_rank}_{self.req_id}'
        self.llm_engine.add_request(
            request_id=new_req_id,
            prompt=dict(
                prompt_token_ids=self.prompt_token_ids,
            ),
            params=sampling_params
        )
        # 还原request_state的状态
        self.recover_request_state(new_req_id)
        # 将scheduler_request的值和状态刷新正确，并恢复scheduler内部的状态
        scheduler_request = self.recover_scheduler_request(new_req_id, sampling_params)
        # 恢复model_runner中的状态和kvCache
        self.recover_model_runner(scheduler_request)


class States:
    config = RolloutRebalanceConfig
    rebalance_engine: RolloutRebalanceEngine
    outputs_cache = []
    graph_batch_sizes = [64, 32, 16, 8, 4]
    sampling_params = None
    reprefill_mode = False


def enable_rollout_rebalance(config=States.config):
    States.config = config

    if not config.enable:
        return

    @hook.before(LLM, '__init__')
    def _(self: LLM, *args, **kwargs) -> None:
        additional_config = kwargs.get('additional_config')
        max_batch_size = additional_config['torchair_graph_config']['graph_batch_sizes'][0]

        # 滤除定义的graph_batch_sizes中超出max_batch_size的部分
        graph_batch_sizes = [max_batch_size]
        for bs in sorted(set(config.graph_batch_sizes), reverse=True):
            if bs < max_batch_size:
                graph_batch_sizes.append(bs)
        States.graph_batch_sizes = graph_batch_sizes
        if config.multi_graph:
            additional_config['torchair_graph_config'].update(
                graph_batch_sizes_init=False,
                use_cached_graph=True,
                graph_batch_sizes=graph_batch_sizes,
            )

    @hook.before(LLM, '_run_engine')
    def _(self: LLM, *args, **kwargs) -> None:
        States.rebalance_engine = RolloutRebalanceEngine(self.llm_engine,
                                                         check_interval=config.check_interval)
        States.outputs_cache = []
        if config.profile:
            States.rebalance_engine.ProfileCache.start(config.profile_interval)
        request_list = list(self.llm_engine.engine_core.engine_core.scheduler.requests.values())
        States.sampling_params = request_list[0].sampling_params

    @hook.after(LLMEngine, 'step')
    def _(self: LLMEngine, step_outputs, *args, **kwargs) -> None:
        States.rebalance_engine.check()
        if step_outputs:
            for request_output in step_outputs[:]:
                if request_output.request_id.startswith('src_rank'):
                    States.outputs_cache.append(request_output)
                    step_outputs.remove(request_output)

    rank_log_info('[RolloutRebalance][Enabled]')
