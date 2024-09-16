# AAD-DCE

Official pytorch implementation of AAD-DCE: An Aggregated Multimodal Attention Mechanism for Early and Late Dynamic Contrast Enhanced Prostate MRI Synthesis.

### Training:
```bash
python3 train.py --dataroot prostate_dataset --name T2_ADC_T1_to_DCE1_DCE2 --gpu_ids 0 --model aad_dce --which_model_netG res_cnn --which_model_netD aad
--which_direction AtoB --lambda_A 100 --dataset_mode aligned --norm batch --pool_size 0 --output_nc 2 --input_nc 3 --loadSize 160 --fineSize 160
--niter 50 --niter_decay 50 --checkpoints_dir checkpoints/ --display_id 0 --lr 0.0002
```
### Testing
```bash
python3 test.py --dataroot prostate_dataset --name T2_ADC_T1_to_DCE1_DCE2 --gpu_ids 0 --model aad_dce --which_model_netG res_cnn 
--dataset_mode aligned --norm batch --phase test --output_nc 2 --input_nc 3 --how_many 10000 --serial_batches --fineSize 160 --loadSize 160 
--results_dir results/ --checkpoints_dir checkpoints/ --which_epoch latest
```
