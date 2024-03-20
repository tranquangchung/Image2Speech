
#folder="output/paper_abslation_study/num_hidden_layers_v2"
#output_dir=$folder"/2layer_v2"
#CUDA_VISIBLE_DEVICES=3 python train_caption_unit.py \
#  --output_dir $output_dir \
#  --config configs/flick8k_audio.yaml \

#loss="l1"
#folder="output/visual_feedback_v3"
#output_dir=$folder"/visual_loss_"$loss"test1"
#CUDA_VISIBLE_DEVICES=0 python train_caption_unit_feedback.py \
#  --visual_feedback $loss \
#  --output_dir $output_dir \
#  --config configs/flick8k_audio.yaml \

#folder="output/dfgan"
#output_dir=$folder"/with_nograd_v3_env_blip"
#CUDA_VISIBLE_DEVICES=2 python train_caption_unit_feedback_v3.py \
#  --output_dir $output_dir \
#  --config configs/flick8k_audio.yaml \

#folder="output/visual_feedback_v4"
#output_dir=$folder"/visual_loss_dfgan_fs_ft" # fs: from scratch, ft: fine-tune
#CUDA_VISIBLE_DEVICES=2 python train_caption_unit_feedback_dfgan.py \
#  --output_dir $output_dir \
#  --config configs/flick8k_audio.yaml \

#folder="output/auditory_feedback_v4"
#output_dir=$folder"/auditory_feedback_ftlm_fs_steplr_08_3layer" # fs: from scratch, ft: fine-tune
#CUDA_VISIBLE_DEVICES=1 python train_caption_unit_feedback_TTSDecoder.py \
#  --output_dir $output_dir \
#  --config configs/flick8k_audio.yaml \

#folder="output/auditory_feedback_v8"
#loss="MSE"
#output_dir=$folder"/auditory_feedback_ftlm_fs_2layer_steplr05_$loss" # fs: from scratch, ft: fine-tune
#CUDA_VISIBLE_DEVICES=0 python train_caption_unit_feedback_TTSDecoder_Attention.py \
#  --output_dir $output_dir \
#  --config configs/flick8k_audio.yaml \
#  --loss_type $loss \


#folder="output/auditory_feedback_v4"
#output_dir=$folder"/auditory_feedback_ftlm_fs_2layer_cosine_transformer" # fs: from scratch, ft: fine-tune
#CUDA_VISIBLE_DEVICES=0 python train_caption_unit_feedback_TTSDecoder_v2.py \
#  --output_dir $output_dir \
#  --config configs/flick8k_audio.yaml \


#folder="output/auditory_feedback_v11"
#loss="MSE"
#output_dir=$folder"/auditory_feedback_ftlm_fs_3layer_cosine_$loss""hidden_feature" # fs: from scratch, ft: fine-tune
#CUDA_VISIBLE_DEVICES=0 python train_caption_unit_feedback_TTSDecoder_AttentionV3.py \
#  --output_dir $output_dir \
#  --config configs/english/3layer/flick8k_audio.yaml \
#  --loss_type $loss \


#folder="output/auditory_feedback_v10"
#loss="MSE"
#output_dir=$folder"/auditory_feedback_ft_fs_2layer_cosine_MSEhidden_feature" # fs: from scratch, ft: fine-tune
#CUDA_VISIBLE_DEVICES=0 python train_caption_unit_feedback_TTSDecoder_AttentionV4.py \
#  --output_dir $output_dir \
#  --config configs/2layer/flick8k_audio.yaml \
#  --loss_type $loss \
