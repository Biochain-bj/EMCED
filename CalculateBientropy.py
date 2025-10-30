import sys, argparse, multiprocessing, pat_util
import pandas as pd
from bitstring import Bits
from bientropy import bien, tbien
from functools import partial

def Bytes_trans(sequence):
    trantab = str.maketrans("TC", "01")
    string = sequence.translate(trantab)
    string = "0b" + string
    ## Methylation Entropy
    ME_value = bien(Bits(string))
    return ME_value

def Calculate_entropy(pat, chrom, start, end):
    _, pat_records, _ = pat.fetch_pat(str(chrom), int(start), int(end), zero_based=False, flank_extend=None)
    '''
    fetch pat records and CT count located in one given genomic region
    return result:
    pat records: a list formated as [[CpGindex_start, pat, pat_count], ...]
    pat records Example:
    11734991, 'CCC', 5
    11734991, 'C..CCTCCCCCCCCT', 1
    11734991, 'CCTTTTTTTTT', 2
    11735007, 'TTTTTTTTTTT', 2
    ...
    '''
    if pat_records is None:
        return 'NA'
    DF = pd.DataFrame(pat_records)
    if DF.empty:
        return 'NA'
    ## filter insert size: indel / less than 3 CpGs / more than 32 CpGs
    ## ADD: CpGs number / bientropy
    DF = DF[(DF[1].str.len() < 33) & (DF[1].str.len() > 2)]	### length filter
    DF = DF[~DF[1].str.contains(r"[.]")]	### indel filter
    DF['Entropy'] = DF[1].apply(Bytes_trans)
    p = sum(DF[2] * DF['Entropy']) / sum(DF[2]) if sum(DF[2]) > 0 else 'NA'
    s = sum(DF[2])
    return p

def entropyAggregate(MethBin, pat_file):
    sampleid = pat_file.split('/')[-1].split('.')[0]
    pat = pat_util.PAT(pat_file, '', 'hg19')
    MethBin[sampleid] = MethBin.apply(lambda x:Calculate_entropy(pat, x['chrom'], x['start'], x['end']), axis=1)
    return MethBin[['marker_index',sampleid]]

def main():
    parser = argparse.ArgumentParser(description='Methylation BioEntropy of One Insert Size')
    parser.add_argument('-m', '--pat_meta', help='patfile metadata', type=str)
    parser.add_argument('-t', '--target_file', help='target region file without header', type=str)
    parser.add_argument('-c', '--cores', help='threads need of multiprocessing', type=int, default=15)
    parser.add_argument('-o', '--outfile', help='outfile', type=str)
    args = parser.parse_args()

    pat_meta = pd.read_csv(args.pat_meta,sep='\t',header=0)
    MethBin = pd.read_csv(args.target_file,sep='\t',header=None,names=['chrom','start','end','marker_index'])
    MethBin = MethBin[MethBin['chrom'].isin(['X','Y'])==False]
    pool = multiprocessing.Pool(args.cores)
    partial_entropyAggregate = partial(entropyAggregate, MethBin)
    results = pool.map_async(partial_entropyAggregate, list(pat_meta['pat_file'])).get()
    pool.close()
    pool.join()
    merged_df = results[0]
    for tmpdf in results[1:]:
        merged_df = pd.merge(merged_df, tmpdf, on='marker_index')
    merged_df.to_csv(args.outfile, sep='\t',index=False)

if __name__ == "__main__":
    main()
