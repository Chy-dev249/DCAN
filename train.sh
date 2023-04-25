# D->A one shot
# da_dis_para
python train_da_dis_para.py --gpu 1 --source_path /home/changhuiying/DCAN-master/data/list/office_20/dslr_20.txt --target_path /home/changhuiying/DCAN-master/data/list/office_20/amazon_20.txt --test_path /home/changhuiying/DCAN-master/data/list/office_20/amazon_20.txt --task da --output_path snapshot/da/uda_dis_da/
python train_da_dis_para.py --gpu 1 --source_path /home/changhuiying/DCAN-master/data/list/office_cida/dslr_10_oneshot_addAmazon.txt --target_path /home/changhuiying/DCAN-master/data/list/office_cida/amazon_20_oneshot.txt --test_path /home/changhuiying/DCAN-master/data/list/office_cida/amazon_20_oneshot.txt --task da --output_path snapshot/da/dca_dis_da/
python train_da_dis_para.py --gpu 1 --source_path /home/changhuiying/DCAN-master/data/list/office_cida/dslr_10_oneshot_addAmazon.txt --target_path /home/changhuiying/DCAN-master/data/list/office_cida/amazon_20_oneshot.txt --test_path /home/changhuiying/DCAN-master/data/list/office_cida/amazon_20_oneshot.txt --task da --sampler cls_balance --output_path snapshot/da/dca_dis_cb_da/
python train_da_dis_para.py --gpu 1 --source_path /home/changhuiying/DCAN-master/data/list/office_cida/dslr_10_oneshot_addAmazon.txt --target_path /home/changhuiying/DCAN-master/data/list/office_cida/amazon_20_oneshot.txt --test_path /home/changhuiying/DCAN-master/data/list/office_cida/amazon_20_oneshot.txt --task da --sampler scp_d --output_path snapshot/da/dca_dis_scpd_da/

# da_adv_para
python train_da_adv_para.py --gpu 1 --source_path /home/changhuiying/DCAN-master/data/list/office_20/dslr_20.txt --target_path /home/changhuiying/DCAN-master/data/list/office_20/amazon_20.txt --test_path /home/changhuiying/DCAN-master/data/list/office_20/amazon_20.txt --task da --output_path snapshot/da/uda_dis_da/
python train_da_adv_para.py --gpu 1 --source_path /home/changhuiying/DCAN-master/data/list/office_cida/dslr_10_oneshot_addAmazon.txt --target_path /home/changhuiying/DCAN-master/data/list/office_cida/amazon_20_oneshot.txt --test_path /home/changhuiying/DCAN-master/data/list/office_cida/amazon_20_oneshot.txt --task da --output_path snapshot/da/dca_dis_da/
python train_da_adv_para.py --gpu 1 --source_path /home/changhuiying/DCAN-master/data/list/office_cida/dslr_10_oneshot_addAmazon.txt --target_path /home/changhuiying/DCAN-master/data/list/office_cida/amazon_20_oneshot.txt --test_path /home/changhuiying/DCAN-master/data/list/office_cida/amazon_20_oneshot.txt --task da --sampler cls_balance --output_path snapshot/da/dca_dis_cb_da/
python train_da_adv_para.py --gpu 2 --source_path /home/changhuiying/DCAN-master/data/list/office_cida/dslr_10_oneshot_addAmazon.txt --target_path /home/changhuiying/DCAN-master/data/list/office_cida/amazon_20_oneshot.txt --test_path /home/changhuiying/DCAN-master/data/list/office_cida/amazon_20_oneshot.txt --task da --sampler scp_d --output_path snapshot/da/dca_dis_scpd_da/



# D->W one shot
# da_dis_para
python train_da_dis_para.py --gpu 1 --source_path /home/changhuiying/DCAN-master/data/list/office_20/dslr_20.txt --target_path /home/changhuiying/DCAN-master/data/list/office_20/webcam_20.txt --test_path /home/changhuiying/DCAN-master/data/list/office_20/webcam_20.txt --task dw --output_path snapshot/dw/uda_dis_dw/
python train_da_dis_para.py --gpu 1 --source_path /home/changhuiying/DCAN-master/data/list/office_cida/dslr_10_oneshot_addWebcam.txt --target_path /home/changhuiying/DCAN-master/data/list/office_cida/webcam_20_oneshot.txt --test_path /home/changhuiying/DCAN-master/data/list/office_cida/webcam_20_oneshot.txt --task dw --output_path snapshot/dw/dca_dis_dw/
python train_da_dis_para.py --gpu 1 --source_path /home/changhuiying/DCAN-master/data/list/office_cida/dslr_10_oneshot_addWebcam.txt --target_path /home/changhuiying/DCAN-master/data/list/office_cida/webcam_20_oneshot.txt --test_path /home/changhuiying/DCAN-master/data/list/office_cida/webcam_20_oneshot.txt --task dw --sampler cls_balance --output_path snapshot/dw/dca_dis_cb_dw/
python train_da_dis_para.py --gpu 1 --source_path /home/changhuiying/DCAN-master/data/list/office_cida/dslr_10_oneshot_addWebcam.txt --target_path /home/changhuiying/DCAN-master/data/list/office_cida/webcam_20_oneshot.txt --test_path /home/changhuiying/DCAN-master/data/list/office_cida/webcam_20_oneshot.txt --task dw --sampler scp_d --output_path snapshot/dw/dca_dis_scpd_dw/

# da_adv_para
python train_da_adv_para.py --gpu 1 --source_path /home/changhuiying/DCAN-master/data/list/office_20/dslr_20.txt --target_path /home/changhuiying/DCAN-master/data/list/office_20/webcam_20.txt --test_path /home/changhuiying/DCAN-master/data/list/office_20/webcam_20.txt --task dw --output_path snapshot/dw/uda_dis_dw/
python train_da_adv_para.py --gpu 1 --source_path /home/changhuiying/DCAN-master/data/list/office_cida/dslr_10_oneshot_addWebcam.txt --target_path /home/changhuiying/DCAN-master/data/list/office_cida/webcam_20_oneshot.txt --test_path /home/changhuiying/DCAN-master/data/list/office_cida/webcam_20_oneshot.txt --task dw --output_path snapshot/dw/dca_dis_dw/
python train_da_adv_para.py --gpu 1 --source_path /home/changhuiying/DCAN-master/data/list/office_cida/dslr_10_oneshot_addWebcam.txt --target_path /home/changhuiying/DCAN-master/data/list/office_cida/webcam_20_oneshot.txt --test_path /home/changhuiying/DCAN-master/data/list/office_cida/webcam_20_oneshot.txt --task dw --sampler cls_balance --output_path snapshot/dw/dca_dis_cb_dw/
python train_da_adv_para.py --gpu 1 --source_path /home/changhuiying/DCAN-master/data/list/office_cida/dslr_10_oneshot_addWebcam.txt --target_path /home/changhuiying/DCAN-master/data/list/office_cida/webcam_20_oneshot.txt --test_path /home/changhuiying/DCAN-master/data/list/office_cida/webcam_20_oneshot.txt --task dw --sampler scp_d --output_path snapshot/dw/dca_dis_scpd_dw/



# A->W one shot
# da_dis_para
python train_da_dis_para.py --gpu 1 --source_path /home/changhuiying/DCAN-master/data/list/office_20/amazon_20.txt --target_path /home/changhuiying/DCAN-master/data/list/office_20/webcam_20.txt --test_path /home/changhuiying/DCAN-master/data/list/office_20/webcam_20.txt --task aw --output_path snapshot/aw/uda_dis_aw/
python train_da_dis_para.py --gpu 1 --source_path /home/changhuiying/DCAN-master/data/list/office_cida/amazon_10_oneshot_addWebcam.txt --target_path /home/changhuiying/DCAN-master/data/list/office_cida/webcam_20_oneshot.txt --test_path /home/changhuiying/DCAN-master/data/list/office_cida/webcam_20_oneshot.txt --task aw --output_path snapshot/aw/dca_dis_aw/
python train_da_dis_para.py --gpu 1 --source_path /home/changhuiying/DCAN-master/data/list/office_cida/amazon_10_oneshot_addWebcam.txt --target_path /home/changhuiying/DCAN-master/data/list/office_cida/webcam_20_oneshot.txt --test_path /home/changhuiying/DCAN-master/data/list/office_cida/webcam_20_oneshot.txt --task aw --sampler cls_balance --output_path snapshot/aw/dca_dis_cb_aw/
python train_da_dis_para.py --gpu 1 --source_path /home/changhuiying/DCAN-master/data/list/office_cida/amazon_10_oneshot_addWebcam.txt --target_path /home/changhuiying/DCAN-master/data/list/office_cida/webcam_20_oneshot.txt --test_path /home/changhuiying/DCAN-master/data/list/office_cida/webcam_20_oneshot.txt --task aw --sampler scp_d --output_path snapshot/aw/dca_dis_scpd_aw/

# da_adv_para
python train_da_adv_para.py --gpu 1 --source_path /home/changhuiying/DCAN-master/data/list/office_20/amazon_20.txt --target_path /home/changhuiying/DCAN-master/data/list/office_20/webcam_20.txt --test_path /home/changhuiying/DCAN-master/data/list/office_20/webcam_20.txt --task aw --output_path snapshot/aw/uda_dis_aw/
python train_da_adv_para.py --gpu 1 --source_path /home/changhuiying/DCAN-master/data/list/office_cida/amazon_10_oneshot_addWebcam.txt --target_path /home/changhuiying/DCAN-master/data/list/office_cida/webcam_20_oneshot.txt --test_path /home/changhuiying/DCAN-master/data/list/office_cida/webcam_20_oneshot.txt --task aw --output_path snapshot/aw/dca_dis_aw/
python train_da_adv_para.py --gpu 1 --source_path /home/changhuiying/DCAN-master/data/list/office_cida/amazon_10_oneshot_addWebcam.txt --target_path /home/changhuiying/DCAN-master/data/list/office_cida/webcam_20_oneshot.txt --test_path /home/changhuiying/DCAN-master/data/list/office_cida/webcam_20_oneshot.txt --task aw --sampler cls_balance --output_path snapshot/aw/dca_dis_cb_aw/
python train_da_adv_para.py --gpu 1 --source_path /home/changhuiying/DCAN-master/data/list/office_cida/amazon_10_oneshot_addWebcam.txt --target_path /home/changhuiying/DCAN-master/data/list/office_cida/webcam_20_oneshot.txt --test_path /home/changhuiying/DCAN-master/data/list/office_cida/webcam_20_oneshot.txt --task aw --sampler scp_d --output_path snapshot/aw/dca_dis_scpd_aw/



# A->D one shot
# da_dis_para
python train_da_dis_para.py --gpu 1 --source_path /home/changhuiying/DCAN-master/data/list/office_20/amazon_20.txt --target_path /home/changhuiying/DCAN-master/data/list/office_20/dslr_20.txt --test_path /home/changhuiying/DCAN-master/data/list/office_20/dslr_20.txt --task ad --output_path snapshot/ad/uda_dis_ad/
python train_da_dis_para.py --gpu 1 --source_path /home/changhuiying/DCAN-master/data/list/office_cida/amazon_10_oneshot_addDslr.txt --target_path /home/changhuiying/DCAN-master/data/list/office_cida/dslr_20_oneshot.txt --test_path /home/changhuiying/DCAN-master/data/list/office_cida/dslr_20_oneshot.txt --task ad --output_path snapshot/ad/dca_dis_ad/
python train_da_dis_para.py --gpu 1 --source_path /home/changhuiying/DCAN-master/data/list/office_cida/amazon_10_oneshot_addDslr.txt --target_path /home/changhuiying/DCAN-master/data/list/office_cida/dslr_20_oneshot.txt --test_path /home/changhuiying/DCAN-master/data/list/office_cida/dslr_20_oneshot.txt --task ad --sampler cls_balance --output_path snapshot/ad/dca_dis_cb_ad/
python train_da_dis_para.py --gpu 1 --source_path /home/changhuiying/DCAN-master/data/list/office_cida/amazon_10_oneshot_addDslr.txt --target_path /home/changhuiying/DCAN-master/data/list/office_cida/dslr_20_oneshot.txt --test_path /home/changhuiying/DCAN-master/data/list/office_cida/dslr_20_oneshot.txt --task ad --sampler scp_d --output_path snapshot/ad/dca_dis_scpd_ad/

# da_adv_para
python train_da_adv_para.py --gpu 1 --source_path /home/changhuiying/DCAN-master/data/list/office_20/amazon_20.txt --target_path /home/changhuiying/DCAN-master/data/list/office_20/dslr_20.txt --test_path /home/changhuiying/DCAN-master/data/list/office_20/dslr_20.txt --task ad --output_path snapshot/ad/uda_dis_ad/
python train_da_adv_para.py --gpu 1 --source_path /home/changhuiying/DCAN-master/data/list/office_cida/amazon_10_oneshot_addDslr.txt --target_path /home/changhuiying/DCAN-master/data/list/office_cida/dslr_20_oneshot.txt --test_path /home/changhuiying/DCAN-master/data/list/office_cida/dslr_20_oneshot.txt --task ad --output_path snapshot/ad/dca_dis_ad/
python train_da_adv_para.py --gpu 1 --source_path /home/changhuiying/DCAN-master/data/list/office_cida/amazon_10_oneshot_addDslr.txt --target_path /home/changhuiying/DCAN-master/data/list/office_cida/dslr_20_oneshot.txt --test_path /home/changhuiying/DCAN-master/data/list/office_cida/dslr_20_oneshot.txt --task ad --sampler cls_balance --output_path snapshot/ad/dca_dis_cb_ad/
python train_da_adv_para.py --gpu 1 --source_path /home/changhuiying/DCAN-master/data/list/office_cida/amazon_10_oneshot_addDslr.txt --target_path /home/changhuiying/DCAN-master/data/list/office_cida/dslr_20_oneshot.txt --test_path /home/changhuiying/DCAN-master/data/list/office_cida/dslr_20_oneshot.txt --task ad --sampler scp_d --output_path snapshot/ad/dca_dis_scpd_ad/




# W->A one shot
# da_dis_para
python train_da_dis_para.py --gpu 1 --source_path /home/changhuiying/DCAN-master/data/list/office_20/webcam_20.txt --target_path /home/changhuiying/DCAN-master/data/list/office_20/amazon_20.txt --test_path /home/changhuiying/DCAN-master/data/list/office_20/amazon_20.txt --task wa --output_path snapshot/wa/uda_dis_wa/
python train_da_dis_para.py --gpu 1 --source_path /home/changhuiying/DCAN-master/data/list/office_cida/webcam_10_oneshot_addAmazon.txt --target_path /home/changhuiying/DCAN-master/data/list/office_cida/amazon_20_oneshot.txt --test_path /home/changhuiying/DCAN-master/data/list/office_cida/amazon_20_oneshot.txt --task wa --output_path snapshot/wa/dca_dis_wa/
python train_da_dis_para.py --gpu 1 --source_path /home/changhuiying/DCAN-master/data/list/office_cida/webcam_10_oneshot_addAmazon.txt --target_path /home/changhuiying/DCAN-master/data/list/office_cida/amazon_20_oneshot.txt --test_path /home/changhuiying/DCAN-master/data/list/office_cida/amazon_20_oneshot.txt --task wa --sampler cls_balance --output_path snapshot/wa/dca_dis_cb_wa/
python train_da_dis_para.py --gpu 1 --source_path /home/changhuiying/DCAN-master/data/list/office_cida/webcam_10_oneshot_addAmazon.txt --target_path /home/changhuiying/DCAN-master/data/list/office_cida/amazon_20_oneshot.txt --test_path /home/changhuiying/DCAN-master/data/list/office_cida/amazon_20_oneshot.txt --task wa --sampler scp_d --output_path snapshot/wa/dca_dis_scpd_wa/

# da_adv_para
python train_da_adv_para.py --gpu 1 --source_path /home/changhuiying/DCAN-master/data/list/office_20/webcam_20.txt --target_path /home/changhuiying/DCAN-master/data/list/office_20/amazon_20.txt --test_path /home/changhuiying/DCAN-master/data/list/office_20/amazon_20.txt --task wa --output_path snapshot/wa/uda_dis_wa/
python train_da_adv_para.py --gpu 1 --source_path /home/changhuiying/DCAN-master/data/list/office_cida/webcam_10_oneshot_addAmazon.txt --target_path /home/changhuiying/DCAN-master/data/list/office_cida/amazon_20_oneshot.txt --test_path /home/changhuiying/DCAN-master/data/list/office_cida/amazon_20_oneshot.txt --task wa --output_path snapshot/wa/dca_dis_wa/
python train_da_adv_para.py --gpu 1 --source_path /home/changhuiying/DCAN-master/data/list/office_cida/webcam_10_oneshot_addAmazon.txt --target_path /home/changhuiying/DCAN-master/data/list/office_cida/amazon_20_oneshot.txt --test_path /home/changhuiying/DCAN-master/data/list/office_cida/amazon_20_oneshot.txt --task wa --sampler cls_balance --output_path snapshot/wa/dca_dis_cb_wa/
python train_da_adv_para.py --gpu 1 --source_path /home/changhuiying/DCAN-master/data/list/office_cida/webcam_10_oneshot_addAmazon.txt --target_path /home/changhuiying/DCAN-master/data/list/office_cida/amazon_20_oneshot.txt --test_path /home/changhuiying/DCAN-master/data/list/office_cida/amazon_20_oneshot.txt --task wa --sampler scp_d --output_path snapshot/wa/dca_dis_scpd_wa/




# W->D one shot
# da_dis_para
python train_da_dis_para.py --gpu 1 --source_path /home/changhuiying/DCAN-master/data/list/office_20/webcam_20.txt --target_path /home/changhuiying/DCAN-master/data/list/office_20/dslr_20.txt --test_path /home/changhuiying/DCAN-master/data/list/office_20/dslr_20.txt --task wd --output_path snapshot/wd/uda_dis_wd/
python train_da_dis_para.py --gpu 1 --source_path /home/changhuiying/DCAN-master/data/list/office_cida/webcam_10_oneshot_addDslr.txt --target_path /home/changhuiying/DCAN-master/data/list/office_cida/dslr_20_oneshot.txt --test_path /home/changhuiying/DCAN-master/data/list/office_cida/dslr_20_oneshot.txt --task wd --output_path snapshot/wd/dca_dis_wd/
python train_da_dis_para.py --gpu 1 --source_path /home/changhuiying/DCAN-master/data/list/office_cida/webcam_10_oneshot_addDslr.txt --target_path /home/changhuiying/DCAN-master/data/list/office_cida/dslr_20_oneshot.txt --test_path /home/changhuiying/DCAN-master/data/list/office_cida/dslr_20_oneshot.txt --task wd --sampler cls_balance --output_path snapshot/wd/dca_dis_cb_wd/
python train_da_dis_para.py --gpu 1 --source_path /home/changhuiying/DCAN-master/data/list/office_cida/webcam_10_oneshot_addDslr.txt --target_path /home/changhuiying/DCAN-master/data/list/office_cida/dslr_20_oneshot.txt --test_path /home/changhuiying/DCAN-master/data/list/office_cida/dslr_20_oneshot.txt --task wd --sampler scp_d --output_path snapshot/wd/dca_dis_scpd_wd/

# da_adv_para
python train_da_adv_para.py --gpu 1 --source_path /home/changhuiying/DCAN-master/data/list/office_20/webcam_20.txt --target_path /home/changhuiying/DCAN-master/data/list/office_20/dslr_20.txt --test_path /home/changhuiying/DCAN-master/data/list/office_20/dslr_20.txt --task wd --output_path snapshot/wd/uda_dis_wd/
python train_da_adv_para.py --gpu 1 --source_path /home/changhuiying/DCAN-master/data/list/office_cida/webcam_10_oneshot_addDslr.txt --target_path /home/changhuiying/DCAN-master/data/list/office_cida/dslr_20_oneshot.txt --test_path /home/changhuiying/DCAN-master/data/list/office_cida/dslr_20_oneshot.txt --task wd --output_path snapshot/wd/dca_dis_wd/
python train_da_adv_para.py --gpu 1 --source_path /home/changhuiying/DCAN-master/data/list/office_cida/webcam_10_oneshot_addDslr.txt --target_path /home/changhuiying/DCAN-master/data/list/office_cida/dslr_20_oneshot.txt --test_path /home/changhuiying/DCAN-master/data/list/office_cida/dslr_20_oneshot.txt --task wd --sampler cls_balance --output_path snapshot/wd/dca_dis_cb_wd/
python train_da_adv_para.py --gpu 1 --source_path /home/changhuiying/DCAN-master/data/list/office_cida/webcam_10_oneshot_addDslr.txt --target_path /home/changhuiying/DCAN-master/data/list/office_cida/dslr_20_oneshot.txt --test_path /home/changhuiying/DCAN-master/data/list/office_cida/dslr_20_oneshot.txt --task wd --sampler scp_d --output_path snapshot/wd/dca_dis_scpd_wd/
