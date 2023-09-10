#!/bin/bash

# ./evaluate.sh 2>&1 | tee "/path/to/Sparkles/models/evaluate.log"

# This function will be called when Ctrl+C is pressed
cleanup() {
    # Kill all child processes
    pkill -P $$
}

# Set the trap
trap "cleanup" SIGINT SIGTERM

# run evaluation simultaneously on 8 GPUs
echo "Evaluating"

set_root="--sparkles_root /path/to/Sparkles/"

start=$(date +%s)

(
start_A=$(date +%s)
CUDA_VISIBLE_DEVICES=0 python evaluate.py ${set_root} --cfg-path eval_configs/sparkles_eval.yaml --num-beams 1 --inference --gpu-id 0 --dataset NLVR2 --data-from 0 --data-to 38 &
CUDA_VISIBLE_DEVICES=1 python evaluate.py ${set_root} --cfg-path eval_configs/sparkles_eval.yaml --num-beams 1 --inference --gpu-id 0 --dataset NLVR2 --data-from 38 --data-to 76 &
CUDA_VISIBLE_DEVICES=2 python evaluate.py ${set_root} --cfg-path eval_configs/sparkles_eval.yaml --num-beams 1 --inference --gpu-id 0 --dataset NLVR2 --data-from 76 --data-to 114 &
CUDA_VISIBLE_DEVICES=3 python evaluate.py ${set_root} --cfg-path eval_configs/sparkles_eval.yaml --num-beams 1 --inference --gpu-id 0 --dataset NLVR2 --data-from 114 --data-to 150 &
wait
CUDA_VISIBLE_DEVICES=0 python evaluate.py ${set_root} --cfg-path eval_configs/sparkles_eval.yaml --num-beams 1 --merge-results --dataset NLVR2
end_A=$(date +%s)
runtime_A=$((end_A-start_A))
echo "NLVR2 evaluation time: $runtime_A seconds"
) &

(
start_B=$(date +%s)
CUDA_VISIBLE_DEVICES=0 python evaluate.py ${set_root} --cfg-path eval_configs/sparkles_eval.yaml --num-beams 1 --inference --gpu-id 0 --dataset BISON --data-from 0 --data-to 38 &
CUDA_VISIBLE_DEVICES=1 python evaluate.py ${set_root} --cfg-path eval_configs/sparkles_eval.yaml --num-beams 1 --inference --gpu-id 0 --dataset BISON --data-from 38 --data-to 76 &
CUDA_VISIBLE_DEVICES=2 python evaluate.py ${set_root} --cfg-path eval_configs/sparkles_eval.yaml --num-beams 1 --inference --gpu-id 0 --dataset BISON --data-from 76 --data-to 114 &
CUDA_VISIBLE_DEVICES=3 python evaluate.py ${set_root} --cfg-path eval_configs/sparkles_eval.yaml --num-beams 1 --inference --gpu-id 0 --dataset BISON --data-from 114 --data-to 150 &
wait
CUDA_VISIBLE_DEVICES=0 python evaluate.py ${set_root} --cfg-path eval_configs/sparkles_eval.yaml --num-beams 1 --merge-results --dataset BISON
end_B=$(date +%s)
runtime_B=$((end_B-start_B))
echo "BISON evaluation time: $runtime_B seconds"
) &

(
start_C=$(date +%s)
CUDA_VISIBLE_DEVICES=4 python evaluate.py ${set_root} --cfg-path eval_configs/sparkles_eval.yaml --num-beams 1 --inference --gpu-id 0 --dataset SparklesEval --data-from 0 --data-to 19 &
CUDA_VISIBLE_DEVICES=4 python evaluate.py ${set_root} --cfg-path eval_configs/sparkles_eval.yaml --num-beams 1 --inference --gpu-id 0 --dataset SparklesEval --data-from 19 --data-to 38 &
CUDA_VISIBLE_DEVICES=5 python evaluate.py ${set_root} --cfg-path eval_configs/sparkles_eval.yaml --num-beams 1 --inference --gpu-id 0 --dataset SparklesEval --data-from 38 --data-to 57 &
CUDA_VISIBLE_DEVICES=5 python evaluate.py ${set_root} --cfg-path eval_configs/sparkles_eval.yaml --num-beams 1 --inference --gpu-id 0 --dataset SparklesEval --data-from 57 --data-to 76 &
CUDA_VISIBLE_DEVICES=6 python evaluate.py ${set_root} --cfg-path eval_configs/sparkles_eval.yaml --num-beams 1 --inference --gpu-id 0 --dataset SparklesEval --data-from 76 --data-to 95 &
CUDA_VISIBLE_DEVICES=6 python evaluate.py ${set_root} --cfg-path eval_configs/sparkles_eval.yaml --num-beams 1 --inference --gpu-id 0 --dataset SparklesEval --data-from 95 --data-to 114 &
CUDA_VISIBLE_DEVICES=7 python evaluate.py ${set_root} --cfg-path eval_configs/sparkles_eval.yaml --num-beams 1 --inference --gpu-id 0 --dataset SparklesEval --data-from 114 --data-to 133 &
CUDA_VISIBLE_DEVICES=7 python evaluate.py ${set_root} --cfg-path eval_configs/sparkles_eval.yaml --num-beams 1 --inference --gpu-id 0 --dataset SparklesEval --data-from 133 --data-to 150 &
wait
CUDA_VISIBLE_DEVICES=0 python evaluate.py ${set_root} --cfg-path eval_configs/sparkles_eval.yaml --num-beams 1 --merge-results --dataset SparklesEval
end_C=$(date +%s)
runtime_C=$((end_C-start_C))
echo "SparklesEval evaluation time: $runtime_C seconds"
) &


(
echo "Evaluating the annotations of SparklesEval (sparkles_evaluation_sparkleseval_annotations)"
start_C=$(date +%s)
python evaluate.py ${set_root} --inference --dataset SparklesEval --data-from 0 --data-to 38 &
python evaluate.py ${set_root} --inference --dataset SparklesEval --data-from 38 --data-to 76 &
python evaluate.py ${set_root} --inference --dataset SparklesEval --data-from 76 --data-to 114 &
python evaluate.py ${set_root} --inference --dataset SparklesEval --data-from 114 --data-to 150 &
wait
python evaluate.py ${set_root} --merge-results --dataset SparklesEval
end_C=$(date +%s)
runtime_C=$((end_C-start_C))
echo "SparklesEval annotation evaluation time: $runtime_C seconds"
) &


wait

end=$(date +%s)
runtime=$((end-start))

echo "Overall evaluation time: $runtime seconds"