#cython: language_level=3
import pandas as pd
import base_util
import cython

def assem_convert(in_patfile: str, out_preffix: str, from_version: str, to_version: str):
	assert from_version in ['hg19', 'hg38'] and to_version in ['hg19', 'hg38'] and from_version != to_version, 'Only allowed to set "to_version" in ["hg19", "hg38"], and [from_version != to_version]'
	index2index = base_util.COORDNATE_INDEX2INDEX(f'{from_version}To{to_version}')
	n_fragment: cython.int = 0
	n_unmap: cython.int = 0
	n_rearrange: cython.int = 0
	items: list
	CpGindex: list
	with open(in_patfile, 'r') as f_read, open(out_preffix + '.out', 'w') as f_out, open(out_preffix + '.unmap', 'w') as f_umap:
		for line in f_read:
			n_fragment += 1
			items = line.strip().split()
			CpGindex = index2index.convert_coordinate(items[0], int(items[1]), int(items[2]))
			if len(CpGindex) == 0:
				n_unmap += 1
				f_umap.write(f'# unmap:\n' + '\t'.join(items) + '\n')
				continue
			if int(items[2]) - int(items[1]) + 1 > len(CpGindex) or not pd.Index(CpGindex).is_monotonic_increasing or CpGindex[-1] - CpGindex[0] + 1 > len(CpGindex):
				n_rearrange += 1
				f_umap.write(f'# rearrange: converted as [{", ".join([str(pos) for pos in CpGindex])}]\n' + '\t'.join(items) + '\n')
				continue
			f_out.write('\t'.join([items[0], str(CpGindex[0]), str(CpGindex[-1]), items[-2], items[-1]]) + '\n')
	return [n_fragment, n_unmap, n_rearrange]