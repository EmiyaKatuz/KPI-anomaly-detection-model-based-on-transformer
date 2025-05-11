export CUDA_VISIBLE_DEVICES=0

#1.SMD数据集: (K=4, len_map=16,anomaly_ratio=0.7)
#训练:
python main_OT.py --k=4  --num_proto 16 --len_map 6   --mode train --anomaly_ratio 0.7 --dataset SMD --data_path dataset/SMD --input_c 38    --output_c 38
#测试:
python main_OT.py --k 4 --num_proto 16 --len_map 6   --mode test --anomaly_ratio 0.7 --dataset SMD --data_path dataset/SMD --input_c 38    --output_c 38

#训练:
python main_OT.py --k=4  --num_proto 16 --len_map 8   --mode train --anomaly_ratio 0.7 --dataset SMD --data_path dataset/SMD --input_c 38    --output_c 38
#测试:
python main_OT.py --k 4 --num_proto 16 --len_map 8   --mode test --anomaly_ratio 0.7 --dataset SMD --data_path dataset/SMD --input_c 38    --output_c 38

#训练:
python main_OT.py --k=4  --num_proto 16 --len_map 10   --mode train --anomaly_ratio 0.7 --dataset SMD --data_path dataset/SMD --input_c 38    --output_c 38
#测试:
python main_OT.py --k 4 --num_proto 16 --len_map 10  --mode test --anomaly_ratio 0.7 --dataset SMD --data_path dataset/SMD --input_c 38    --output_c 38

#训练:
python main_OT.py --k=4  --num_proto 16 --len_map 12   --mode train --anomaly_ratio 0.7 --dataset SMD --data_path dataset/SMD --input_c 38    --output_c 38
#测试:
python main_OT.py --k 4 --num_proto 16 --len_map 12   --mode test --anomaly_ratio 0.7 --dataset SMD --data_path dataset/SMD --input_c 38    --output_c 38

#训练:
python main_OT.py --k=4  --num_proto 16 --len_map 16   --mode train --anomaly_ratio 0.7 --dataset SMD --data_path dataset/SMD --input_c 38    --output_c 38
#测试:
python main_OT.py --k 4 --num_proto 16 --len_map 16   --mode test --anomaly_ratio 0.7 --dataset SMD --data_path dataset/SMD --input_c 38    --output_c 38

#---------------------------------------------------------------------------------------------------------------------------------------------------
#2.MSL数据集：(K=3,len_map=16,anomaly_ratio=0.8)
#训练:
python main_OT.py --k 3 --num_proto 12 --len_map 6   --mode train --anomaly_ratio 0.8 --dataset MSL --data_path dataset/MSL --input_c 55    --output_c 55
#测试:
python main_OT.py --k 3 --num_proto 12 --len_map 6   --mode test --anomaly_ratio 0.8 --dataset MSL --data_path dataset/MSL --input_c 55 --output_c 55

#训练:
python main_OT.py --k 3 --num_proto 12 --len_map 8   --mode train --anomaly_ratio 0.8 --dataset MSL --data_path dataset/MSL --input_c 55    --output_c 55
#测试:
python main_OT.py --k 3 --num_proto 12 --len_map 8   --mode test --anomaly_ratio 0.8 --dataset MSL --data_path dataset/MSL --input_c 55 --output_c 55

#训练:
python main_OT.py --k 3 --num_proto 12 --len_map 10   --mode train --anomaly_ratio 0.8 --dataset MSL --data_path dataset/MSL --input_c 55    --output_c 55
#测试:
python main_OT.py --k 3 --num_proto 12 --len_map 10   --mode test --anomaly_ratio 0.8 --dataset MSL --data_path dataset/MSL --input_c 55 --output_c 55

#训练:
python main_OT.py --k 3 --num_proto 12 --len_map 12   --mode train --anomaly_ratio 0.8 --dataset MSL --data_path dataset/MSL --input_c 55    --output_c 55
#测试:
python main_OT.py --k 3 --num_proto 12 --len_map 12   --mode test --anomaly_ratio 0.8 --dataset MSL --data_path dataset/MSL --input_c 55 --output_c 55

#训练:
python main_OT.py --k 3 --num_proto 12 --len_map 16   --mode train --anomaly_ratio 0.8 --dataset MSL --data_path dataset/MSL --input_c 55    --output_c 55
#测试:
python main_OT.py --k 3 --num_proto 12 --len_map 16   --mode test --anomaly_ratio 0.8 --dataset MSL --data_path dataset/MSL --input_c 55 --output_c 55

#---------------------------------------------------------------------------------------------------------------------------------------------------
#3.PSM数据集: (K=1,len_map=10,anomaly_ratio=0.6)
#
#
#---------------------------------------------------------------------------------------------------------------------------------------------------
#
#4.SMAP数据集: (K=3,num_proto=10,anomaly_ratio=0.7)
#训练:
python main_OT.py --k 3 --num_proto 10 --len_map 6    --mode train --anomaly_ratio 0.7 --dataset SMAP --data_path dataset/SMAP --input_c 25    --output_c 25
#测试:
python main_OT.py --k 3 --num_proto 10 --len_map 6    --mode test --anomaly_ratio 0.7 --dataset SMAP --data_path dataset/SMAP --input_c 25    --output_c 25

#训练:
python main_OT.py --k 3 --num_proto 10 --len_map 8    --mode train --anomaly_ratio 0.7 --dataset SMAP --data_path dataset/SMAP --input_c 25    --output_c 25
#测试:
python main_OT.py --k 3 --num_proto 10 --len_map 8    --mode test --anomaly_ratio 0.7 --dataset SMAP --data_path dataset/SMAP --input_c 25    --output_c 25

#训练:
python main_OT.py --k 3 --num_proto 10 --len_map 10    --mode train --anomaly_ratio 0.7 --dataset SMAP --data_path dataset/SMAP --input_c 25    --output_c 25
#测试:
python main_OT.py --k 3 --num_proto 10 --len_map 10    --mode test --anomaly_ratio 0.7 --dataset SMAP --data_path dataset/SMAP --input_c 25    --output_c 25

#训练:
python main_OT.py --k 3 --num_proto 10 --len_map 12    --mode train --anomaly_ratio 0.7 --dataset SMAP --data_path dataset/SMAP --input_c 25    --output_c 25
#测试:
python main_OT.py --k 3 --num_proto 10 --len_map 12    --mode test --anomaly_ratio 0.7 --dataset SMAP --data_path dataset/SMAP --input_c 25    --output_c 25

#训练:
python main_OT.py --k 3 --num_proto 10 --len_map 16    --mode train --anomaly_ratio 0.7 --dataset SMAP --data_path dataset/SMAP --input_c 25    --output_c 25
#测试:
python main_OT.py --k 3 --num_proto 10 --len_map 16    --mode test --anomaly_ratio 0.7 --dataset SMAP --data_path dataset/SMAP --input_c 25    --output_c 25



#---------------------------------------------------------------------------------------------------------------------------------------------------
#5.SWaT数据集: (K=2,num_proto=6,anomaly_ratio=0.5)
#训练:
python main_OT.py --k 2 --num_proto 6 --len_map 6     --mode train --anomaly_ratio 0.5 --dataset SWaT --data_path dataset/SWaT --input_c 51    --output_c 51
#测试:
python main_OT.py --k 2 --num_proto 6 --len_map 6     --mode test --anomaly_ratio 0.5 --dataset SWaT --data_path dataset/SWaT --input_c 51    --output_c 51

#训练:
python main_OT.py --k 2 --num_proto 6 --len_map 8     --mode train --anomaly_ratio 0.5 --dataset SWaT --data_path dataset/SWaT --input_c 51    --output_c 51
#测试:
python main_OT.py --k 2 --num_proto 6 --len_map 8     --mode test --anomaly_ratio 0.5 --dataset SWaT --data_path dataset/SWaT --input_c 51    --output_c 51

#训练:
python main_OT.py --k 2 --num_proto 6 --len_map 10    --mode train --anomaly_ratio 0.5 --dataset SWaT --data_path dataset/SWaT --input_c 51    --output_c 51
#测试:
python main_OT.py --k 2 --num_proto 6 --len_map 10     --mode test --anomaly_ratio 0.5 --dataset SWaT --data_path dataset/SWaT --input_c 51    --output_c 51

#训练:
python main_OT.py --k 2 --num_proto 6 --len_map 12     --mode train --anomaly_ratio 0.5 --dataset SWaT --data_path dataset/SWaT --input_c 51    --output_c 51
#测试:
python main_OT.py --k 2 --num_proto 6 --len_map 12     --mode test --anomaly_ratio 0.5 --dataset SWaT --data_path dataset/SWaT --input_c 51    --output_c 51

#训练:
python main_OT.py --k 2 --num_proto 6 --len_map 16     --mode train --anomaly_ratio 0.5 --dataset SWaT --data_path dataset/SWaT --input_c 51    --output_c 51
#测试:
python main_OT.py --k 2 --num_proto 6 --len_map 16     --mode test --anomaly_ratio 0.5 --dataset SWaT --data_path dataset/SWaT --input_c 51    --output_c 51
