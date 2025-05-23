export CUDA_VISIBLE_DEVICES=0

python main_OT.py --k=4  --num_proto 16 --len_map 6   --mode train --anomaly_ratio 0.7 --dataset SMD --data_path dataset/SMD --input_c 38    --output_c 38

python main_OT.py --k 4 --num_proto 16 --len_map 6   --mode test --anomaly_ratio 0.7 --dataset SMD --data_path dataset/SMD --input_c 38    --output_c 38


python main_OT.py --k=4  --num_proto 16 --len_map 8   --mode train --anomaly_ratio 0.7 --dataset SMD --data_path dataset/SMD --input_c 38    --output_c 38

python main_OT.py --k 4 --num_proto 16 --len_map 8   --mode test --anomaly_ratio 0.7 --dataset SMD --data_path dataset/SMD --input_c 38    --output_c 38


python main_OT.py --k=4  --num_proto 16 --len_map 10   --mode train --anomaly_ratio 0.7 --dataset SMD --data_path dataset/SMD --input_c 38    --output_c 38

python main_OT.py --k 4 --num_proto 16 --len_map 10  --mode test --anomaly_ratio 0.7 --dataset SMD --data_path dataset/SMD --input_c 38    --output_c 38


python main_OT.py --k=4  --num_proto 16 --len_map 12   --mode train --anomaly_ratio 0.7 --dataset SMD --data_path dataset/SMD --input_c 38    --output_c 38

python main_OT.py --k 4 --num_proto 16 --len_map 12   --mode test --anomaly_ratio 0.7 --dataset SMD --data_path dataset/SMD --input_c 38    --output_c 38


python main_OT.py --k=4  --num_proto 16 --len_map 16   --mode train --anomaly_ratio 0.7 --dataset SMD --data_path dataset/SMD --input_c 38    --output_c 38

python main_OT.py --k 4 --num_proto 16 --len_map 16   --mode test --anomaly_ratio 0.7 --dataset SMD --data_path dataset/SMD --input_c 38    --output_c 38


python main_OT.py --k 3 --num_proto 12 --len_map 6   --mode train --anomaly_ratio 0.8 --dataset MSL --data_path dataset/MSL --input_c 55    --output_c 55

python main_OT.py --k 3 --num_proto 12 --len_map 6   --mode test --anomaly_ratio 0.8 --dataset MSL --data_path dataset/MSL --input_c 55 --output_c 55


python main_OT.py --k 3 --num_proto 12 --len_map 8   --mode train --anomaly_ratio 0.8 --dataset MSL --data_path dataset/MSL --input_c 55    --output_c 55

python main_OT.py --k 3 --num_proto 12 --len_map 8   --mode test --anomaly_ratio 0.8 --dataset MSL --data_path dataset/MSL --input_c 55 --output_c 55


python main_OT.py --k 3 --num_proto 12 --len_map 10   --mode train --anomaly_ratio 0.8 --dataset MSL --data_path dataset/MSL --input_c 55    --output_c 55

python main_OT.py --k 3 --num_proto 12 --len_map 10   --mode test --anomaly_ratio 0.8 --dataset MSL --data_path dataset/MSL --input_c 55 --output_c 55


python main_OT.py --k 3 --num_proto 12 --len_map 12   --mode train --anomaly_ratio 0.8 --dataset MSL --data_path dataset/MSL --input_c 55    --output_c 55

python main_OT.py --k 3 --num_proto 12 --len_map 12   --mode test --anomaly_ratio 0.8 --dataset MSL --data_path dataset/MSL --input_c 55 --output_c 55


python main_OT.py --k 3 --num_proto 12 --len_map 16   --mode train --anomaly_ratio 0.8 --dataset MSL --data_path dataset/MSL --input_c 55    --output_c 55

python main_OT.py --k 3 --num_proto 12 --len_map 16   --mode test --anomaly_ratio 0.8 --dataset MSL --data_path dataset/MSL --input_c 55 --output_c 55


python main_OT.py --k 3 --num_proto 10 --len_map 6    --mode train --anomaly_ratio 0.7 --dataset SMAP --data_path dataset/SMAP --input_c 25    --output_c 25

python main_OT.py --k 3 --num_proto 10 --len_map 6    --mode test --anomaly_ratio 0.7 --dataset SMAP --data_path dataset/SMAP --input_c 25    --output_c 25


python main_OT.py --k 3 --num_proto 10 --len_map 8    --mode train --anomaly_ratio 0.7 --dataset SMAP --data_path dataset/SMAP --input_c 25    --output_c 25

python main_OT.py --k 3 --num_proto 10 --len_map 8    --mode test --anomaly_ratio 0.7 --dataset SMAP --data_path dataset/SMAP --input_c 25    --output_c 25


python main_OT.py --k 3 --num_proto 10 --len_map 10    --mode train --anomaly_ratio 0.7 --dataset SMAP --data_path dataset/SMAP --input_c 25    --output_c 25

python main_OT.py --k 3 --num_proto 10 --len_map 10    --mode test --anomaly_ratio 0.7 --dataset SMAP --data_path dataset/SMAP --input_c 25    --output_c 25


python main_OT.py --k 3 --num_proto 10 --len_map 12    --mode train --anomaly_ratio 0.7 --dataset SMAP --data_path dataset/SMAP --input_c 25    --output_c 25

python main_OT.py --k 3 --num_proto 10 --len_map 12    --mode test --anomaly_ratio 0.7 --dataset SMAP --data_path dataset/SMAP --input_c 25    --output_c 25


python main_OT.py --k 3 --num_proto 10 --len_map 16    --mode train --anomaly_ratio 0.7 --dataset SMAP --data_path dataset/SMAP --input_c 25    --output_c 25

python main_OT.py --k 3 --num_proto 10 --len_map 16    --mode test --anomaly_ratio 0.7 --dataset SMAP --data_path dataset/SMAP --input_c 25    --output_c 25


python main_OT.py --k 2 --num_proto 6 --len_map 6     --mode train --anomaly_ratio 0.5 --dataset SWaT --data_path dataset/SWaT --input_c 51    --output_c 51

python main_OT.py --k 2 --num_proto 6 --len_map 6     --mode test --anomaly_ratio 0.5 --dataset SWaT --data_path dataset/SWaT --input_c 51    --output_c 51


python main_OT.py --k 2 --num_proto 6 --len_map 8     --mode train --anomaly_ratio 0.5 --dataset SWaT --data_path dataset/SWaT --input_c 51    --output_c 51

python main_OT.py --k 2 --num_proto 6 --len_map 8     --mode test --anomaly_ratio 0.5 --dataset SWaT --data_path dataset/SWaT --input_c 51    --output_c 51


python main_OT.py --k 2 --num_proto 6 --len_map 10    --mode train --anomaly_ratio 0.5 --dataset SWaT --data_path dataset/SWaT --input_c 51    --output_c 51

python main_OT.py --k 2 --num_proto 6 --len_map 10     --mode test --anomaly_ratio 0.5 --dataset SWaT --data_path dataset/SWaT --input_c 51    --output_c 51


python main_OT.py --k 2 --num_proto 6 --len_map 12     --mode train --anomaly_ratio 0.5 --dataset SWaT --data_path dataset/SWaT --input_c 51    --output_c 51

python main_OT.py --k 2 --num_proto 6 --len_map 12     --mode test --anomaly_ratio 0.5 --dataset SWaT --data_path dataset/SWaT --input_c 51    --output_c 51


python main_OT.py --k 2 --num_proto 6 --len_map 16     --mode train --anomaly_ratio 0.5 --dataset SWaT --data_path dataset/SWaT --input_c 51    --output_c 51

python main_OT.py --k 2 --num_proto 6 --len_map 16     --mode test --anomaly_ratio 0.5 --dataset SWaT --data_path dataset/SWaT --input_c 51    --output_c 51
