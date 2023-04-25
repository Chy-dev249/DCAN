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