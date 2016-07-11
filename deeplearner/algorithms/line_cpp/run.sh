./reconstruct -train input.edges -output dense_train.txt -depth 2 -k-max 1000
./line -train dense_train.txt -output vec_1st_wo_norm.txt -binary 1 -size 128 -order 1 -negative 5 -samples 10 -threads 30
./line -train dense_train.txt -output vec_2nd_wo_norm.txt -binary 1 -size 128 -order 2 -negative 5 -samples 10 -threads 30
./normalize -input vec_1st_wo_norm.txt -output vec_1st.txt -binary 1
./normalize -input vec_2nd_wo_norm.txt -output vec_2nd.txt -binary 1
./concatenate -input1 vec_1st.txt -input2 vec_2nd.txt -output vec_all.txt -text 1
