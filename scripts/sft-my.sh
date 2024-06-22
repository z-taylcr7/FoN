
if [ -z "${BASH_VERSION}" ]; then
        echo "Please use bash to run this script." >&1
        exit 1
fi

set -x

SCRIPT_DIR="$(cd "$(dirname "$0")" &>/dev/null && pwd)"
ROOT_DIR="$(dirname "${SCRIPT_DIR}")"
export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

MODEL_NAME_OR_PATH="/root/hf_7B"
OUTPUT_DIR="${ROOT_DIR}/output/sft-ra+regen"
ZERO_STAGE=3
TRAIN_DATASETS="alpaca"
MAX_LENGTH=512
NUM_NODES=1
NUM_GPUS=8
MASTER_PORT=13579
MASTER_ADDR="127.0.0.1"
RESPONSE_ONLY="False"
PROMPT_TYPE="none"
OFFLOAD="False"
EPOCHS=10
while [[ "$#" -gt 0 ]]; do
        arg="$1"
        shift
        case "${arg}" in
                --model_name_or_path)
                        MODEL_NAME_OR_PATH="$1"
                        shift
                        ;;
                --model_name_or_path=*)
                        MODEL_NAME_OR_PATH="${arg#*=}"
                        ;;
                --output_dir)
                        OUTPUT_DIR="$1"
                        shift
                        ;;
                --output_dir=*)
                        OUTPUT_DIR="${arg#*=}"
                        ;;
                --zero_stage)
                        ZERO_STAGE="$1"
                        shift
                        ;;
                --zero_stage=*)
                        ZERO_STAGE="${arg#*=}"
                        ;;
                --train_datasets)
                        TRAIN_DATASETS="$1"
                        shift
                        ;;
                --train_datasets=*)
                        TRAIN_DATASETS="${arg#*=}"
                        ;;
                --max_length)
                        MAX_LENGTH="$1"
                        shift
                        ;;
                --max_length=*)
                        MAX_LENGTH="${arg#*=}"
                        ;;
                --num_nodes)
                        NUM_NODES="$1"
                        shift
                        ;;
                --num_nodes=*)
                        NUM_NODES="${arg#*=}"
                        ;;
                --num_gpus)
                        NUM_GPUS="$1"
                        shift
                        ;;
                --num_gpus=*)
                        NUM_GPUS="${arg#*=}"
                        ;;
                --master_port)
                        MASTER_PORT="$1"
                        shift
                        ;;
                --master_port=*)
                        MASTER_PORT="${arg#*=}"
                        ;;
                --response_only)
                        RESPONSE_ONLY="$1"
                        shift
                        ;;
                --response_only=*)
                        RESPONSE_ONLY="${arg#*=}"
                        ;;
                --prompt_type)
                        PROMPT_TYPE="$1"
                        shift
                        ;;
                --prompt_type=*)
                        PROMPT_TYPE="${arg#*=}"
                        ;;
                --offload)
                        OFFLOAD="$1"
                        shift
                        ;;
                --offload=*)
                        OFFLOAD="${arg#*=}"
                        ;;
                --epochs)
                        EPOCHS="$1"
                        shift
                        ;;
                --epochs=*)
                        EPOCHS="${arg#*=}"
                        ;;
                *)
                        echo "Unknown parameter passed: $1" >&2
                        exit 1
                        ;;
        esac
done

mkdir -p "${OUTPUT_DIR}"
OUTPUT_DIR="$(cd "${OUTPUT_DIR}" &>/dev/null && pwd)"
if [[ ! -f "${OUTPUT_DIR}/.gitignore" ]]; then
        echo '*' >"${OUTPUT_DIR}/.gitignore"
fi

exec 1> >(tee "${OUTPUT_DIR}/stdout.log" >&1) 2> >(tee "${OUTPUT_DIR}/stderr.log" >&2)

# you can assign different gpus for this script by modifing the --include
deepspeed --include localhost:6,7 --master_port $MASTER_PORT \
        --module safe_rlhf.finetune \
        --train_datasets gsm8k_regen \
        --model_name_or_path  /data/ckpts/mistralai/Mistral-7B-v0.1 \
        --max_length 2048 \
        --epochs 3 \
        --per_device_train_batch_size 4\
        --per_device_eval_batch_size 4 \
        --gradient_accumulation_steps 8\
        --save_interval 100000000 \
        --learning_rate 2e-5 \
        --lr_scheduler_type cosine \
        --weight_decay 0.1 \
        --seed 42 \
        --output_dir "${OUTPUT_DIR}" \
        --log_type wandb \
        --log_project Safe-RLHF-SFT \
        --zero_stage 3 \
        --bf16 True \
        --tf32 True \
        --gradient_checkpointing \
        --offload none \
