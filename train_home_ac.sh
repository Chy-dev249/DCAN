#ca scp_d
#ca
python train_da_adv_para.py --gpu 3 --source_path /home/huiying/uda/dcan/data/list/home/Art_40_oneshot_addClipart.txt --target_path /home/huiying/uda/dcan/data/list/home/Clipart_65_oneshot.txt --test_path /home/huiying/uda/dcan/data/list/home/Clipart_65_oneshot.txt --task ac --sampler scp_d --output_path snapshot/dca_advpara_scpd/ --data_set home --max_temp 1000 --min_temp 100
python train_da_dis_para.py --gpu 3 --source_path /home/huiying/uda/dcan/data/list/home/Art_40_oneshot_addClipart.txt --target_path /home/huiying/uda/dcan/data/list/home/Clipart_65_oneshot.txt --test_path /home/huiying/uda/dcan/data/list/home/Clipart_65_oneshot.txt --task ac --sampler scp_d --output_path snapshot/dca_dispara_scpd/ --data_set home --max_temp 1000 --min_temp 100