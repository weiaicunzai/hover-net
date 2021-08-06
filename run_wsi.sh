export CUDA_VISIBLE_DEVICES=0

#--model_path='/data/by/tmp/hover_net/pretrained/hovernet_original_consep_notype_pytorch.tar' \
ls /data/by/tmp/hover_net/pretrained/hovernet_original_consep_notype_pytorch.tar
python run_infer.py \
--gpu='0,1' \
--type_info_path=type_info.json \
--batch_size=16 \
--model_mode=original \
--model_path='/data/by/tmp/hover_net/logs/00/net_epoch=48.tar' \
--nr_inference_workers=8 \
--nr_post_proc_workers=16 \
wsi \
--input_dir=/data/by/tmp/hover_net/samples/input \
--output_dir=/data/by/tmp/hover_net/samples/out \
#--input_mask_dir=/data/by/tmp/hover_net/samples/msk \
#--save_thumb \
#--save_mask

#export CUDA_VISIBLE_DEVICES=0
#ls /data/by/tmp/hover_net/pretrained/hovernet_original_consep_notype_pytorch.tar
#python run_infer.py \
#--gpu='0' \
#--type_info_path=type_info.json \
#--batch_size=16 \
#--model_mode=original \
#--model_path='/data/by/tmp/hover_net/logs/00/net_epoch=48.tar' \
#--nr_inference_workers=8 \
#--nr_post_proc_workers=16 \
#tile \
#--input_dir='/data/by/tmp/hover_net/samples/input' \
#--output_dir=/data/by/tmp/hover_net/samples/out \
#--mem_usage=0.1 \
##--draw_dot \
##--save_qupath \
#
#
#
#
#
##--model_path=../pretrained/hovernet_fast_pannuke_type_tf2pytorch.tar \
##--model_mode=fast \
##--input_mask_dir=dataset/sample_wsis/msk/ \
##--nr_types=6 \
#
