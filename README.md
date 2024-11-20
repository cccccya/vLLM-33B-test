# vLLM 测试
Qwen2.5-32B-Instruct-GPTQ-Int8  
GPU服务器：NVIDIA A100-SXM4-80GB*1，CPU8核，内存14G，系统盘200G
## test0
在8K、40K、80K的prompt长度下测qps上限。输入全为prefix，处理速度最快，最极限
```shell
# Server
python -m vllm.entrypoints.openai.api_server \
--model /root/Qwen2.5-32B-Instruct-GPTQ-Int8 \
--disable-log-requests \
--max-model-len 90000 \
--rope-scaling '{"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 32768}' \
--gpu-memory-utilization 0.9 \
--enable-chunked-prefill \
--enable-prefix-caching

# Client 不指定req-rate
python benchmark_serving.py \
--model /root/Qwen2.5-32B-Instruct-GPTQ-Int8 \
--dataset-name random \
--ignore-eos \
--random-input-len 0 \
--random-output-len 150 \
--random-prefix-len 8192 \
--num-prompts 50 \
--random-range-ratio 1 > ../../vLLM-33B-test/test0/8k.txt
```
最终测得  
8K：0.75QPS（num-prompts=1000）  
20K：0.29QPS（num-prompts=50）   
40K：0.14QPS（num-prompts=100）  
80K：0.06QPS（num-prompts=50）  
## test1
在8K、20K、80K的输入长度下，测开关prefix-cache的效果。输入全为prefix-cache。QPS设置为0.25，与6B统一。
```shell
# Server
python -m vllm.entrypoints.openai.api_server \
--model /root/Qwen2.5-32B-Instruct-GPTQ-Int8 \
--disable-log-requests \
--max-model-len 90000 \
--rope-scaling '{"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 32768}' \
--gpu-memory-utilization 0.9 \
--enable-chunked-prefill \
--enable-prefix-caching

# Client
python benchmark_serving.py \
--model /root/Qwen2.5-32B-Instruct-GPTQ-Int8 \
--dataset-name random \
--request-rate 0.25 \
--ignore-eos \
--random-input-len 0 \
--random-output-len 150 \
--random-prefix-len 8192 \
--num-prompts 50 \
--random-range-ratio 1 > ../../vLLM-33B-test/test1/8k_withkvcache.txt
```
最终测得
| len | cache | TTFP | TTFT/len |
|  ----  | ---- | ---- | ---- |
| 8k | √ | 2087.23 | 0.2548 |
| 8k | × |  |
| 20k | √ | 8131.16 | 0.3970 |
| 20k | × |  |
| 80k | √ |  |
| 80k | × |  |
# test2
TODO
# test3
TODO
# test4
TODO
