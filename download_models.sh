mkdir checkpoints 
cd checkpoints 
# mkdir avsr 
# cd avsr 
# wget https://dl.fbaipublicfiles.com/avhubert/model/lrs3_vox/avsr/base_noise_pt_noise_ft_30h.pt 
# wget https://dl.fbaipublicfiles.com/avhubert/model/lrs3_vox/avsr/base_noise_pt_noise_ft_433h.pt 
# wget https://dl.fbaipublicfiles.com/avhubert/model/lrs3_vox/avsr/large_noise_pt_noise_ft_30h.pt 
# wget https://dl.fbaipublicfiles.com/avhubert/model/lrs3_vox/avsr/large_noise_pt_noise_ft_433h.pt
# cd ..

# mkdir vsr 
# cd vsr
# wget https://dl.fbaipublicfiles.com/avhubert/model/lrs3/vsr/base_lrs3_30h.pt 
# wget https://dl.fbaipublicfiles.com/avhubert/model/lrs3/vsr/base_lrs3_433h.pt
# wget https://dl.fbaipublicfiles.com/avhubert/model/lrs3/vsr/large_lrs3_30h.pt
# wget https://dl.fbaipublicfiles.com/avhubert/model/lrs3/vsr/large_lrs5_433h.pt 
# wget https://dl.fbaipublicfiles.com/avhubert/model/lrs3_vox/vsr/base_vox_30h.pt 
# wget https://dl.fbaipublicfiles.com/avhubert/model/lrs3_vox/vsr/base_vox_433h.pt 
# wget https://dl.fbaipublicfiles.com/avhubert/model/lrs3_vox/vsr/large_vox_30h.pt
# wget https://dl.fbaipublicfiles.com/avhubert/model/lrs3_vox/vsr/large_vox_433h.pt 
# wget https://dl.fbaipublicfiles.com/avhubert/model/lrs3_vox/vsr/self_large_vox_30h.pt 
# wget https://dl.fbaipublicfiles.com/avhubert/model/lrs3_vox/vsr/self_large_vox_433h.pt
# cd ..

mkdir pretrain 
cd pretrain
wget https://dl.fbaipublicfiles.com/avhubert/model/lrs3/clean-pretrain/base_lrs3_iter4.pt
wget https://dl.fbaipublicfiles.com/avhubert/model/lrs3/clean-pretrain/base_lrs3_iter5.pt 
wget https://dl.fbaipublicfiles.com/avhubert/model/lrs3/clean-pretrain/large_lrs3_iter5.pt
wget https://dl.fbaipublicfiles.com/avhubert/model/lrs3_vox/clean-pretrain/base_vox_iter4.pt
wget https://dl.fbaipublicfiles.com/avhubert/model/lrs3_vox/clean-pretrain/base_vox_iter5.pt 
wget https://dl.fbaipublicfiles.com/avhubert/model/lrs3_vox/clean-pretrain/large_vox_iter5.pt
wget https://dl.fbaipublicfiles.com/avhubert/model/lrs3_vox/noise-pretrain/base_vox_iter5.pt 
wget https://dl.fbaipublicfiles.com/avhubert/model/lrs3_vox/noise-pretrain/large_vox_iter5.pt 
cd ../..


