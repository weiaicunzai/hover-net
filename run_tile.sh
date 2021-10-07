#python run_infer.py \
#--gpu='1' \
#--nr_types=6 \
#--type_info_path=type_info.json \
#--batch_size=64 \
#--model_mode=fast \
#--model_path=../pretrained/hovernet_fast_pannuke_type_tf2pytorch.tar \
#--nr_inference_workers=8 \
#--nr_post_proc_workers=16 \
#tile \
#--input_dir=dataset/sample_tiles/imgs/ \
#--output_dir=dataset/sample_tiles/pred/ \
#--mem_usage=0.1 \
#--draw_dot \
#--save_qupath
#--model_path='/data/by/tmp/hover_net/pretrained/hovernet_original_consep_notype_pytorch.tar' \
#--model_path='/data/by/tmp/hover_net/logs/01/net_epoch=36.tar' \
#--model_path='/data/by/tmp/hover_net/logs/00/net_epoch=48.tar' \
#--model_path='/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/HoverNet/net_epoch=36.tar' \
export CUDA_VISIBLE_DEVICES=0
ls pretrained/hovernet_original_consep_notype_pytorch.tar
python -u run_infer.py \
--nr_types=6 \
--gpu='0' \
--type_info_path=type_info.json \
--batch_size=64  \
--model_mode=original \
--model_path='/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/HoverNet/combined_withtypes/net_epoch=50.tar' \
--nr_inference_workers=4 \
--nr_post_proc_workers=4 \
tile \
--input_dir='/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/CRC/SliceImage/' \
--output_dir='/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/CRC/Json/SliceJson' \
#--input_dir='/data/smb/syh/colon_dataset/CRC_Dataset' \
#--input_dir='/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/BACH/Images/test' \
#--output_dir='/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/BACH/Json/withtypes/test' \
#--mem_usage=0.6 \
#--input_dir='/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/BACH/Images/train' \
#--output_dir='/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/BACH/Json/withtypes/train' \
#--input_dir='/data/smb/syh/colon_dataset/CoNSeP/Train/Images/' \
#--output_dir='/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/CoNSeP/Json' \
#--input_dir='/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/TCGA_Prostate/Images/images/' \
#--output_dir='/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/TCGA_Prostate/Json/json_withtypes' \
#--input_dir='/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/BACH/Images/' \
#--output_dir='/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/BACH/Json/' \
#--input_dir='/data/smb/syh/PycharmProjects/CGC-Net/data_prostate_tcga/data/images/' \
#--output_dir='/data/smb/syh/PycharmProjects/CGC-Net/data_prostate_tcga/data/json' \
#--output_dir='tmp' \
#--input_dir='/home/baiyu/tmp/hover_net/prostate_images' \
#--input_dir='/home/baiyu/test_can_be_del3/' \
#--output_dir='/data/smb/syh/PycharmProjects/CGC-Net/data_su/raw/Extended_CRC/mask/' \
#--mem_usage=0.6 \
#--input_dir='/data/by/tmp/HGIN/test_can_be_del3/fold_1/1_normal/' \
#--output_dir=/data/by/tmp/hover_net/samples/out \
#--input_dir='/data/by/tmp/hover_net/samples/input' \
#--draw_dot \
#--save_qupath \
#--input_dir='/data/by/tmp/HGIN/test_can_be_del3/fold_1/1_normal/' \
#--output_dir='/data/smb/syh/PycharmProjects/CGC-Net/data_su/raw/Extended_CRC/mask/fold_1/1_normal' \
