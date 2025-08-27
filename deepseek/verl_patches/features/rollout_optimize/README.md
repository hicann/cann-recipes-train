# RL On-Policy 推理场景的序列级均衡调度引擎

## 1. 简介

### 1.1 背景
RLHF的Rollout阶段面临典型的“木桶效应”：由于输入Prompt所生成的响应（Response）长度存在长尾分布，少数极长的生成任务会拖慢整个DP组的进展。这使得处理短序列的节点在完成计算后不得不进入长时间的闲置等待，造成算力浪费。长尾问题优化的本质是RL训练系统的负载均衡，对此，我们针对单轮推理的同步场景，优化目标是提升进入长尾状态后的推理效率。


### 1.2 解决方案
本优化的核心目标是在On Policy场景中，针对部分rollout提前结束导致各rank间的负载不均时，对未结束的rollout进行负载均衡的策略分析和重调度，从而提升计算资源的利用率和长尾状态下的推理效率。

前置依赖：
vllm_asencd 091上在torchair_graph_config中提供了use_cached_graph和graph_batch_sizes的能力，支持提前配置多档位BS的图，并随着剩余Seq减少时自动匹配最小BS的图进行推理。

本方案中包含以下关键功能实现：
1. Rebalance条件检测与调度策略生成；
2. Request(SEQ)级的数据搬迁与恢复（包含对应的kvCache）；
3. Rollout后的结果还原；


### 1.3 实验结果
我们在A3集群256Die上进行了如下实验，发现本方案开启后单轮推理耗时从6200s左右优化到约2300s，性能收益达57%~62%左右。

实验配置如下：
* 模型：DeepSeekV3;
* 数据集：open-r1/OpenR1-Math-220K;
* data.train_batch_size=512;
* data.max_response_length=32768;
* actor_rollout_ref.rollout.n=16;
* TP=2; DP=128;

性能收益主要来自于单个step的TPOT性能的差距，默认场景下的TPOT会从125ms上升到200ms，而通过使能Rebalance并配合多档位编图，能在1~2K推理长度时就快速将推理档位降低，让单个step的TPOT降低到60ms的量级，在长尾场景下，性能差距被持续放大。

### 1.4 具体实现
以下的详细代码均位于`rollout_rebalance.py`文件中：
#### 全局状态感知
```python 
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
```

#### 基于BatchSize档位预设的最大档位最小化均衡算法
```python 
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
```

#### 序列请求的跨Rank发送与接收（含KvCache）
```python 
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
```
#### KvCache搬迁与恢复
```python
        # request级的kvCache采集
        self.layers_kv_cache_blocks = []
        for cache_block_index in range(len(self.global_kv_caches[0])):
            self.layers_kv_cache_blocks.append(
                torch.stack([layer[cache_block_index][request_block_table] for layer in self.global_kv_caches])
            )

        # kvCache还原
        for layer_index, layer_caches in enumerate(self.model_runner.kv_caches):
            reload_indexes = list(range(len(new_block_ids)))
            for i, cache_block in enumerate(self.layers_kv_cache_blocks):
                layer_caches[i][new_block_ids] = cache_block[layer_index][reload_indexes]
```

## 2. 使用说明
### 2.1 初始化配置
在`verl/workers/megatron_workers.py`文件中的`ActorRolloutRefWokrer.init_model`方法开头追加了以下代码，通过环境变量`ROLLOUT_REBALANCE_ENABLE=1`使能本特性功能：
```python
@register(dispatch_mode=Dispatch.ONE_TO_ALL)
def init_model(self):
    if os.getenv("ROLLOUT_REBALANCE_ENABLE", "0") != "0":
        from features.rollout_optimize.rollout_rebalance import enable_rollout_rebalance
        enable_rollout_rebalance()
```

### 2.2 配置项介绍
可以直接在`config.py`中修改配置，或将以下配置写入verl启动的yaml中，在2.1节所写位置进行配置提取并传入enable_rollout_rebalance方法。
```python
class RolloutRebalanceConfig:
    enable = True  # RolloutRebalance特性总开关
    check_interval = 1000  # 间隔多少个step进行一次rebalance检查

    multi_graph = True  # 是否开启多档位编图，如果关闭，rebalance依然会按预编图的档位做均衡调度，但是不会形成明显的性能收益
    graph_batch_sizes = [64, 32, 16, 8, 4]  # 预编图的档位设置

    profile = True  # 是否打印过程中的性能数据
    profile_interval = 100   # 打印间隔步长
```


