python nnet/separate.py exp/conv_tasnet/adam_libri_shuf_8000_00 --input evals/infer_mix.scp --gpu 0 > adam_libri_shuf_8000_00_1586213903.log 2>&1 &
python nnet/separate.py exp/conv_tasnet/first_test_asa_timit_vctk_libri_18th --input evals/infer_mix.scp --gpu 0 > first_test_asa_timit_vctk_libri_18th_1586215009.log 2>&1 &
python nnet/separate.py exp/conv_tasnet/libri_all_02 --input evals/infer_mix.scp --gpu 0 > libri_all_02_1586213897.log 2>&1 &
python nnet/separate.py exp/conv_tasnet/rec_norm_00_shuf --input evals/infer_mix.scp --gpu 0 > rec_norm_00_shuf_1586215373.log 2>&1 &




