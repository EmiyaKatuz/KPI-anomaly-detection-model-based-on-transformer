export CUDA_VISIBLE_DEVICES=0

python main_OT.py --k 2 --num_proto 6 --len_map 8     --mode test --anomaly_ratio 0.5 --dataset SWaT --data_path dataset/SWaT --input_c 51    --output_c 51

python main_OT.py --k 2 --num_proto 6 --len_map 8     --mode test --anomaly_ratio 0.6 --dataset SWaT --data_path dataset/SWaT --input_c 51    --output_c 51

python main_OT.py --k 2 --num_proto 6 --len_map 8     --mode test --anomaly_ratio 0.7 --dataset SWaT --data_path dataset/SWaT --input_c 51    --output_c 51

python main_OT.py --k 2 --num_proto 6 --len_map 8     --mode test --anomaly_ratio 0.8 --dataset SWaT --data_path dataset/SWaT --input_c 51    --output_c 51

python main_OT.py --k 2 --num_proto 6 --len_map 8     --mode test --anomaly_ratio 0.9 --dataset SWaT --data_path dataset/SWaT --input_c 51    --output_c 51