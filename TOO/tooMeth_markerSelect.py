import argparse
import os
import time
import numpy as np
import pandas as pd
from scipy.stats import kruskal
from statsmodels.stats.multitest import multipletests
from scikit_posthocs import posthoc_dunn
from multiprocessing import Pool


# 定义一个 Timer 类用于测量时间
class Timer:
    def __init__(self, message="Elapsed time"):
        self.message = message

    def __enter__(self):
        self.start_time = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = time.time()
        elapsed_time = end_time - self.start_time
        print(f"{self.message}: {elapsed_time:.4f} seconds")

# Step 1: Kruskal-Wallis Test
def kruskal_test(var, sample_type):
    p_value = 1
    unique_types = np.unique(sample_type)    
    grouped_data = [var[sample_type == t] for t in unique_types]
    
    if any(len(np.unique(group)) > 1 for group in grouped_data):
        p_value = kruskal(*grouped_data).pvalue
    return p_value

def process_file_chunked(filename, chunk_size=20):
    meth_var = []
    sample_type = []
    data_chunks = []
    
    with open(filename, 'r') as file:
        header = file.readline().strip().split('\t')
        meth_var = header[3:]  # Assumes first 3 columns are ['sample_name', 'sample_type', ...]
        
        for line in file:
            fields = line.strip().split('\t')
            sample_type.append(fields[1])
            data_chunks.append([float(x) for x in fields[3:]])  # Convert relevant columns to float
    
    return np.array(sample_type), np.array(data_chunks).T, meth_var  # Transpose data_chunks to get columns

# Parallel processing for Kruskal-Wallis test
def parallel_kruskal(meth_var, data_matrix, sample_type, num_threads=10):
    with Pool(num_threads) as pool:
        kw_res_list = pool.starmap(kruskal_test, [(data_matrix[i, :], sample_type) for i in range(len(meth_var))])
    return kw_res_list

def dunn_test_wrapper(index, filtered_data, sample_type):
    var = filtered_data[index, :]
    unique_types = np.unique(sample_type)
    grouped_data = [var[sample_type == t] for t in unique_types]
    df_res0 = posthoc_dunn(list(grouped_data), p_adjust='bonferroni')

    dunn_res0 = []
    for i in range(0, 6):
        for j in range(i+1, 7):
            dunn_res0.append(df_res0.iloc[i, j])

    pair_test = []
    for i in range(6):
        for j in range(i+1, 7):
            pair_test.append(unique_types[i]+"_"+unique_types[j])

    dunn_dict = dict(zip(pair_test, dunn_res0))
    dunn_deduped_df = pd.DataFrame(dunn_dict, index=[0])

    return dunn_deduped_df

# 定义 main 函数来解析参数并执行相关步骤
def main():
    parser = argparse.ArgumentParser(description="Kruskal-Wallis and Dunn's tests")
    parser.add_argument('--input_file', required=True, help="Path to the input file")
    parser.add_argument('--output_dir', required=True, help="Directory to save the output files")
    parser.add_argument('--chunk_size', type=int, default=20, help="Chunk size for reading the file")
    parser.add_argument('--num_threads', type=int, default=15, help="Number of threads for parallel processing")
    
    args = parser.parse_args()

    # 加载数据
    sample_type, data_matrix, meth_var = process_file_chunked(args.input_file, args.chunk_size)

    # Step 1: Running Kruskal-Wallis test and adjusting p-values
    with Timer("Kruskal-Wallis Test Time"):
        kw_res_list = parallel_kruskal(meth_var, data_matrix, sample_type, args.num_threads)

        kw_res = pd.DataFrame({'var_name': meth_var, 'pvalue': kw_res_list})
        kw_res['qvalue'] = multipletests(kw_res['pvalue'], method='fdr_bh')[1]
        kw_res.to_csv(os.path.join(args.output_dir, 'kw_res.txt'), sep='\t', index=False)

    # Step 2: Dunn's Test
    # Step 2.1: Filtering features with qvalue < 0.005
    kw_filter = kw_res[kw_res['qvalue'] < 0.01]
    filtered_vars = list(set(kw_filter['var_name']))
    filtered_data = data_matrix[np.isin(meth_var, filtered_vars)]

    # Running Dunn's test for the filtered variables
    with Timer("Dunn's Test Time"):
        with Pool(args.num_threads) as pool:
            dunn_res_list = pool.starmap(dunn_test_wrapper, [(i, filtered_data, sample_type) for i in range(len(kw_filter))])
        # 合并所有结果的 DataFrame
        final_dunn_df = pd.concat(dunn_res_list, ignore_index=True)
        final_dunn_df.insert(0, "marker_index", filtered_vars)
        final_dunn_df.to_csv(os.path.join(args.output_dir, 'dunnTest_res.txt'), sep='\t', index=False)

if __name__ == "__main__":
    main()
