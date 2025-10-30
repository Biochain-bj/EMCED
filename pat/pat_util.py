import os, sys, io, gzip, re, math, pysam, subprocess, logging, multiprocessing, uuid, shutil, base_util, cbam2pat, cpat_assemconvert, tempfile
import numpy as np
import pandas as pd
from typing import Optional, Literal

__version__ = '1.1'

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.DEBUG)

def bam2pat(bam_file: str, outdir: str, sample_name: str, chrom: Optional[str]=None, assem_version: str='hg19', overwrite: bool=False, threads: int=32):
	'''
	generate pat file from bam file
	if chrom==None, extract genome-wise pat records
	if chrom==chrX, extract chromosome-wise pat records
	'''
	if chrom is None:
		temp_dir = os.path.join(outdir, f'{sample_name}.PID{os.getpid()}.{str(uuid.uuid4())[:8]}')
		os.mkdir(temp_dir)
		# get all chromosomes present in the bam file header, only primary chromosomes
		bam_chroms = [chrom for chrom in subprocess.check_output(f'samtools idxstats {bam_file} | cut -f1', shell=True).decode(encoding = "utf-8").strip().split('\n')[:-1] if re.match(r'^(chr)?[\dXY]+$', chrom)]
		# generate pat file for each chrom parallelly
		for chrom in bam_chroms:
			bam2pat(bam_file, temp_dir, sample_name, chrom=chrom, assem_version=assem_version, threads=threads)
		# 
		# concate pat files and index
		concat_cmd = f'echo -ne "#META\\tRF:{assem_version}\\n" > {outdir}/{sample_name}.pat ' + \
		'&& cat ' + ' '.join([f'{temp_dir}/{sample_name}.{chrom}.header' for chrom in bam_chroms]) + f' >> {outdir}/{sample_name}.pat ' + \
		'&& cat ' + ' '.join([f'{temp_dir}/{sample_name}.{chrom}.pat' for chrom in bam_chroms]) + f' >> {outdir}/{sample_name}.pat ' + \
		f'&& bgzip {outdir}/{sample_name}.pat ' + \
		'&& cat ' + ' '.join([f'{temp_dir}/{sample_name}.{chrom}.invalid' for chrom in bam_chroms]) + f' > {outdir}/{sample_name}.invalid'
		returncode = subprocess.call(concat_cmd, shell=True)
		assert returncode==0, f'Error occurred: {concat_cmd}'
		index_cmd = f'tabix -b 2 -e 3 {outdir}/{sample_name}.pat.gz'
		returncode = subprocess.call(index_cmd, shell=True)
		assert returncode==0, f'Error occurred: {index_cmd}'
		# remove temporary files
		shutil.rmtree(temp_dir)
	else:
		if not os.path.exists(os.path.join(outdir, f'{sample_name}.{chrom}.pat')) or overwrite:
			os.mkdir(f'{outdir}/{chrom}')
			logging.info(f'[bam2pat] [{chrom}] paired-end reads collate')
			# extract reads from bam file within given chrom, then collate paired-end reads, so that paired reads will appear in adjacent lines
			collate_cmd = f'samtools view -h -F 1796 -f 3 -@ 8 {bam_file} {chrom} | samtools collate -@ {threads - 9} -O - {outdir}/{chrom}/collate | samtools view > {outdir}/{chrom}/{sample_name}.{chrom}.sam'
			returncode = subprocess.call(collate_cmd, shell=True)
			assert returncode==0, f'Error occurred: {collate_cmd}'
			# split collate output into fixed size sam files to boost following calculation by parallel
			nr = int(subprocess.check_output(f'wc -l {outdir}/{chrom}/{sample_name}.{chrom}.sam', shell=True).decode(encoding = "utf-8").strip().split(' ')[0])
			nsplit = math.ceil(nr / threads)
			nsplit += nsplit % 2
			split_cmd = f'split -l {nsplit} -d --additional-suffix .split {outdir}/{chrom}/{sample_name}.{chrom}.sam {outdir}/{chrom}/{sample_name}.{chrom}.'
			returncode = subprocess.call(split_cmd, shell=True)
			assert returncode==0, f'Error occurred: {split_cmd}'
			# fetch meth pattern for each paired-end reads
			logging.info(f'[bam2pat] [{chrom}] fetch meth pattern for each paired-end reads')
			results = []
			with multiprocessing.Pool(threads) as pool:
				for sam_file in [f'{outdir}/{chrom}/{filename}' for filename in os.listdir(f'{outdir}/{chrom}') if filename.endswith('split')]:
					results.append(pool.apply_async(cbam2pat.bam2pat, (sam_file, sam_file.replace('split', 'out'), assem_version)))
				pool.close()
				pool.join()
			# count for n_fragment, n_empty, n_novalid, n_invalid
			n_fragment = sum([i.get()[0] for i in results])
			n_empty = sum([i.get()[1] for i in results])
			n_novalid = sum([i.get()[2] for i in results])
			n_invalid = sum([i.get()[3] for i in results])
			logging.info(f'[bam2pat] [{chrom}] finish fetch meth pattern. empty={n_empty/n_fragment}({n_empty}/{n_fragment}), novalid={n_novalid/n_fragment}({n_novalid}/{n_fragment}), invalid={n_invalid/n_fragment}({n_invalid}/{n_fragment})')
			# generate pat.gz file
			logging.info(f'[bam2pat] [{chrom}] sort and count for each pat record, then generate pat file')
			pat_collate_cmd = 'cat ' + ' '.join([f'{outdir}/{chrom}/{filename}' for filename in os.listdir(f'{outdir}/{chrom}') if filename.endswith('out')]) + f" | sort -k2,2n -k3,3 | uniq -c | awk -v OFS='\t' '{{print $2,$3,$3+length($4)-1,$4,$1}}' > {outdir}/{sample_name}.{chrom}.pat " + \
			f"&& awk '{{n+=$NF}} END{{print \"#META\\tSQ:{chrom.replace('chr', '')}\\tNF:\" n}}' {outdir}/{sample_name}.{chrom}.pat > {outdir}/{sample_name}.{chrom}.header " + \
			'&& cat ' + ' '.join([f'{outdir}/{chrom}/{filename}' for filename in os.listdir(f'{outdir}/{chrom}') if filename.endswith('invalid')]) + f' > {outdir}/{sample_name}.{chrom}.invalid'
			returncode = subprocess.call(pat_collate_cmd, shell=True)
			assert returncode==0, f'Error occurred: {pat_collate_cmd}'
			shutil.rmtree(f'{outdir}/{chrom}')
		else:
			logging.info(f'[bam2pat] [{chrom}] file {outdir}/{sample_name}.{chrom}.pat already exists, skipping it')

def pat_subsample(pat_file: str, rate: float, outfile: str, label: Optional[str]=None):
	'''
	random sub-sample fragments from pat file with given rate
	
	it would append given lablel string into the last column
	'''
	assert 0 < rate < 1, 'Only allow to set rate in (0, 1)'
	with io.TextIOWrapper(gzip.open(pat_file, 'rb'), encoding='utf-8') as pat, open(outfile, 'w') as out:
		for line in pat:
			if line.startswith('#'): continue
			items = line.strip().split()
			n_sample = np.random.binomial(int(items[4]), rate)
			if n_sample > 0:
				if label is not None:
					out.write('\t'.join(items[:4]) + f'\t{n_sample}\t{label}\n')
				else:
					out.write('\t'.join(items[:4]) + f'\t{n_sample}\n')

class PAT:
	def __init__(self, pat_file: str, sample_name: Optional[str]=None, assem_version: str='hg19'):
		assert os.path.exists(pat_file), f'No such file: {pat_file}'
		assert assem_version in ['hg19', 'hg38'], f"Only allowed to set 'assem_version' in ['hg19', 'hg38']"
		self.__pat_file = pat_file
		self.__tbx_pat = pysam.TabixFile(self.__pat_file)
		self.__sample_name = sample_name if sample_name is not None else os.path.basename(self.__pat_file).replace('.pat.gz', '')
		self.__assem_version = assem_version
		self.__pos2index = base_util.COORDNATE_POS2INDEX(self.__assem_version)

	def __region_validate(self, chrom: str, start: int, end: int):
		assert re.match(r'^(chr)?[0-9XYM]+', chrom), 'Invalid genomic region, only allowed to given chrom in [(chr)1, ..., (chr)X, (chr)Y, (chr)M]'
		assert start<=end, f'Invalid genomic region, please make sure [start<=end]'

	def fetch_pat(self, chrom: str, start: int, end: int, zero_based: bool=False, flank_extend: Optional[int]=None):
		'''
		fetch pat records and CT count located in one given genomic region
		return two types of result:
		(1) pat records: a list formated as [[CpGindex_start, pat, pat_count], ...]
		(2) CT count: a list formated as [n_C, n_T]
		'''
		# fist of all, convert genomic coordinate of given genomic region into CpGindex coordinate
		if zero_based: start += 1
		self.__region_validate(chrom, start, end)
		CpGindex = self.__pos2index.convert_coordinate(chrom, start, end, interchange='pos2index')
		assert CpGindex[0] is not None, 'Not find CpG site in this genomic region'
		assert flank_extend is None or flank_extend >= 0, 'Only allowed to set flank_extend>=0'
		# fetch pat records and CT count
		if not ( (re.match(r'^chr', chrom) and re.match(r'^chr', self.__tbx_pat.contigs[0])) or (re.match(r'^[0-9XYM]+', chrom) and re.match(r'^[0-9XYM]+', self.__tbx_pat.contigs[0])) ):
			if re.match(r'^chr', self.__tbx_pat.contigs[0]):
				chrom = 'chr' + chrom
			else:
				chrom = re.sub(r'^chr', '', chrom)
		pat_records = []
		CT_count = [0, 0]
		for tbx_row in self.__tbx_pat.fetch(chrom, CpGindex[0]-1, CpGindex[1]):
			items = tbx_row.strip().split()
			if flank_extend is None:
				cur_pat = items[3]
				pat_start = int(items[1])
			else:
				cur_pat = items[3][(max(CpGindex[0] - flank_extend, int(items[1])) - int(items[1])):(min(CpGindex[1] + flank_extend, int(items[2])) - int(items[1]) + 1)]
				pat_start = max(CpGindex[0] - flank_extend, int(items[1]))
			pat_records.append([pat_start, cur_pat, int(items[4])])
			CT_count[0] += cur_pat.count('C') * int(items[4])
			CT_count[1] += cur_pat.count('T') * int(items[4])
		return CpGindex, pat_records, CT_count

	def fetch_CTcount(self, intervals, zero_based: bool=False):
		'''
		fetch CT count for a set of given genomic regions
		'''
		CT_counts = []
		for row in intervals.itertuples():
			pat_records = self.fetch_pat(row.chrom, row.start, row.end, zero_based=zero_based, flank_extend=0)
			CT_counts.append(pat_records[-1] + [pat_records[0][1] - pat_records[0][0] + 1])
		return CT_counts

	def __count_UXM(self, pat_records: list, threshold_site: int=4, threshold_low: float=1/4, threshold_high: float=3/4):
		UXM_count = [0] * 3
		# pat records: a list formated as [[CpGindex_start, pat, pat_count], ...]
		for pat_record in pat_records:
			if pat_record[1].count('C') + pat_record[1].count('T') >= threshold_site:
				meth = pat_record[1].count('C')/(pat_record[1].count('C') + pat_record[1].count('T'))
				if meth <= threshold_low:
					UXM_count[0] += pat_record[2]
				elif meth >= threshold_high:
					UXM_count[2] += pat_record[2]
				else:
					UXM_count[1] += pat_record[2]
		return UXM_count

	def fetch_UXMcount(self, intervals, zero_based: bool=False, threshold_site: int=4, threshold_low: float=1/4, threshold_high: float=3/4, flank_extend: Optional[int]=None):
		'''
		fetch UXM count for a set of given genomic regions
		'''
		UXM_counts = []
		for row in intervals.itertuples():
			UXM_counts.append(self.__count_UXM(self.fetch_pat(row.chrom, row.start, row.end, zero_based=zero_based, flank_extend=flank_extend)[1], threshold_site=threshold_site, threshold_low=threshold_low, threshold_high=threshold_high))
		return UXM_counts

	def pat_vis(self, chrom: str, start: int, end: int, zero_based: bool=False, flank_extend: Optional[int]=None, as_circles: bool=False, sort_by: bool=False):
		'''
		visualize fetched pats in given genomic region
		'''
		if zero_based: start += 1
		CpGindex, pat_records, CT_count = self.fetch_pat(chrom, start, end, flank_extend=flank_extend)
		if sort_by:
			pat_records.sort(key=lambda x: ((x[1].count('C')/(x[1].count('C') + x[1].count('T'))), x[0]))
		print(f'{chrom}:{start}-{end} {end - start + 1}bp, {CpGindex[1] - CpGindex[0] + 1}CpGs:{CpGindex[0]}-{CpGindex[1]}')
		self.__cyclic_print(CpGindex, pat_records, as_circles=as_circles)

	def __cyclic_print(self, CpGindex, pat_records, as_circles=False):
		'''
		cyclic print for given pat_records
		'''
		# get vis_window: CpGindex_start - CpGindex_end
		pat_starts = [pat_rec[0] for pat_rec in pat_records]
		pat_ends = [pat_rec[0] + len(pat_rec[1]) - 1 for pat_rec in pat_records]
		vis_window = [min(pat_starts), max(pat_ends)]
		# convert original pat strings into padded pat strings using numpy array
		pat_intTab = np.zeros((sum([pat_rec[-1] for pat_rec in pat_records]), vis_window[1] - vis_window[0] + 1), dtype=np.int8)
		n_row = 0
		for pat_rec in pat_records:
			pat_ints = [base_util.str2int[l] for l in pat_rec[1]]
			for c in range(pat_rec[-1]):
				# insert read and spaces:
				pat_intTab[n_row, (pat_rec[0]-vis_window[0]):(pat_rec[0]-vis_window[0] + len(pat_rec[1]))] = pat_ints
				pat_intTab[n_row, :(pat_rec[0]-vis_window[0])] = 1                   # before read
				pat_intTab[n_row, (pat_rec[0]-vis_window[0] + len(pat_rec[1])):] = 1 # after read
				n_row += 1
		pat_txtTab = pd.DataFrame(data=pat_intTab).replace(base_util.int2str).values
		region_tag = ' ' * (CpGindex[0] - vis_window[0]) + '+' * (CpGindex[1] - CpGindex[0] + 1)
		txt_color = base_util.color_text('\n'.join([''.join(row) for row in pat_txtTab]), base_util.num2color_dict)
		if as_circles:
			txt_color = re.sub('[CTUXM]', base_util.FULL_CIRCLE, txt_color)
		print(region_tag + '\n' + txt_color)

	def assem_convert(self, to_version: str, outdir: str, chrom: Optional[str]=None, threads: int=32):
		'''
		convert genomic assembly version for pat file from from_version to to_version
		'''
		assert to_version in ['hg19', 'hg38'] and self.__assem_version != to_version, 'Only allowed to set "to_version" in ["hg19", "hg38"], and [assem_version != to_version]'
		if chrom is None:
			temp_dir = f'{outdir}/{self.__sample_name}.PID{os.getpid()}.{str(uuid.uuid4())[:8]}'
			os.mkdir(temp_dir)
			tbx_chroms = subprocess.check_output(f'tabix -l {self.__pat_file}', shell=True).decode(encoding = "utf-8").strip().split('\n')
			for cur_chrom in tbx_chroms:
				self.assem_convert(to_version, temp_dir, cur_chrom, threads=threads)
			concat_cmd = 'cat ' + ' '.join([f'{temp_dir}/{self.__sample_name}.{to_version}.{cur_chrom}.pat.gz' for cur_chrom in tbx_chroms]) + f' > {outdir}/{self.__sample_name}.{to_version}.pat.gz && cat ' + ' '.join([f'{temp_dir}/{self.__sample_name}.{to_version}.{cur_chrom}.unmap' for cur_chrom in tbx_chroms]) + f' > {outdir}/{self.__sample_name}.{to_version}.unmap'
			returncode = subprocess.call(concat_cmd, shell=True)
			assert returncode==0, f'Error occurred: {concat_cmd}'
			index_cmd = f'tabix -b 2 -e 3 {outdir}/{self.__sample_name}.{to_version}.pat.gz'
			returncode = subprocess.call(index_cmd, shell=True)
			assert returncode==0, f'Error occurred: {index_cmd}'
			# remove temporary files
			shutil.rmtree(temp_dir)
		else:
			if not os.path.exists(os.path.join(outdir, f'{self.__sample_name}.{to_version}.{chrom}.pat.gz')):
				os.mkdir(f'{outdir}/{chrom}')
				# extract pat records within given chrom
				logging.info(f'[assem_convert] [{chrom}] convert genomic assembly version from {self.__assem_version} to {to_version}')
				view_cmd = f'tabix {self.__pat_file} {chrom} > {outdir}/{chrom}/{self.__sample_name}.{chrom}.pat'
				# | split -l 100000 -d --additional-suffix .pat - {outdir}/{chrom}/{self.__sample_name}.'
				returncode = subprocess.call(view_cmd, shell=True)
				assert returncode==0, f'Error occurred: {view_cmd}'
				# split collate output into fixed size pat files to boost following calculation by parallel
				nr = int(subprocess.check_output(f'wc -l {outdir}/{chrom}/{self.__sample_name}.{chrom}.pat', shell=True).decode(encoding = "utf-8").strip().split(' ')[0])
				nsplit = math.ceil(nr / threads)
				split_cmd = f'split -l {nsplit} -d --additional-suffix .split {outdir}/{chrom}/{self.__sample_name}.{chrom}.pat {outdir}/{chrom}/{self.__sample_name}.{chrom}.'
				returncode = subprocess.call(split_cmd, shell=True)
				assert returncode==0, f'Error occurred: {split_cmd}'
				results = []
				with multiprocessing.Pool(threads) as pool:
					for pat_file in [f'{outdir}/{chrom}/{filename}' for filename in os.listdir(f'{outdir}/{chrom}') if filename.endswith('split')]:
						results.append(pool.apply_async(cpat_assemconvert.assem_convert, (pat_file, pat_file.replace('.split', ''), self.__assem_version, to_version)))
					pool.close()
					pool.join()
				# count for n_fragment, n_unmap and n_rearrange
				n_fragment = sum([i.get()[0] for i in results])
				n_unmap = sum([i.get()[1] for i in results])
				n_rearrange = sum([i.get()[2] for i in results])
				logging.info(f'[assem_convert] [{chrom}] finish convert. unmap={n_unmap/n_fragment}({n_unmap}/{n_fragment}), rearrange={n_rearrange/n_fragment}({n_rearrange}/{n_fragment})')
				# generate pat.gz file
				logging.info(f'[assem_convert] [{chrom}] sort for pat records, then generate bgziped pat file')
				pat_collate_cmd = 'cat ' + ' '.join([f'{outdir}/{chrom}/{filename}' for filename in os.listdir(f'{outdir}/{chrom}') if filename.endswith('out')]) + f' | sort -k2,2n -k4,4 | bgzip > {outdir}/{self.__sample_name}.{to_version}.{chrom}.pat.gz && cat ' + ' '.join([f'{outdir}/{chrom}/{filename}' for filename in os.listdir(f'{outdir}/{chrom}') if filename.endswith('unmap')]) + f' > {outdir}/{self.__sample_name}.{to_version}.{chrom}.unmap'
				returncode = subprocess.call(pat_collate_cmd, shell=True)
				assert returncode==0, f'Error occurred: {pat_collate_cmd}'
				shutil.rmtree(f'{outdir}/{chrom}')
			else:
				logging.info(f'[assem_convert] [{chrom}] file {outdir}/{self.__sample_name}.{to_version}.{chrom}.pat.gz already exists, skipping it')

	def __calc_alpha_distribution(self, chrom: str, start: int, end: int, zero_based: bool=False, flank_extend: Optional[int]=None, step_len: float=0.05, min_fragment: int=10):
		'''
		for a given genomic region, fetch pat records located within it, then calculate alpha-value distribution in range [0, 1], and step 0.05(default).
		
		the alpha-value distribution defined as: the fraction of fragments (CpG sites no less than 4) whose alpha-value were less than specified cut
		'''
		# fist of all, convert genomic coordinate of given genomic region into CpGindex coordinate
		if zero_based: start += 1
		self.__region_validate(chrom, start, end)
		# pat records: a list formated as [[CpGindex_start, pat, pat_count], ...]
		pat_records = self.fetch_pat(chrom, start, end, flank_extend=flank_extend)[1]
		# alpha records: a list formated as [[alpha, count], ...]
		alpha_records = [[rec[1].count('C')/(rec[1].count('C') + rec[1].count('T')), rec[-1]] for rec in pat_records if rec[1].count('C') + rec[1].count('T')>=4]
		if len(alpha_records) > 0 and sum([rec[-1] for rec in alpha_records])>=min_fragment:
			return list(np.array([sum([rec[-1] for rec in alpha_records if rec[0]<alpha_cut]) for alpha_cut in np.arange(0, 1 + step_len, step_len)])/sum([rec[-1] for rec in alpha_records]))
		else:
			return [np.nan] * len(np.arange(0, 1 + step_len, step_len))

	def fetch_distribution(self, intervals, zero_based: bool=False, flank_extend: Optional[int]=None, step_len: float=0.05, min_fragment: int=10):
		'''
		fetch (reverse) alpha-value distribution for a set of given genomic regions
		
		the alpha-value distribution defined as: the fraction of fragments whose alpha-value were less than specified cut in range [0, 1], and step 0.05
		'''
		alpha_distribution = []
		for row in intervals.itertuples():
			alpha_distribution.append(self.__calc_alpha_distribution(row.chrom, row.start, row.end, zero_based=zero_based, flank_extend=flank_extend, step_len=step_len, min_fragment=min_fragment))
		return alpha_distribution

	def fetch_fragquant(self, chrom: str, start: int, end: int, alpha_cut: float, meth_direct: Literal['hypo', 'hyper'], zero_based: bool=False, flank_extend: Optional[int]=None):
		'''
		fetch fragment quantification for fragments located in given genomic region, which satisfied with alpha <= alpha_cut if meth_direct=='hypo', otherwise, satisfied with alpha >= alpha_cut if meth_direct=='hyper'
		
		return as [fragquant, (valid) frag count]
		'''
		# fist of all, convert genomic coordinate of given genomic region into CpGindex coordinate
		if zero_based: start += 1
		self.__region_validate(chrom, start, end)
		# pat records: a list formated as [[CpGindex_start, pat, pat_count], ...]
		pat_records = self.fetch_pat(chrom, start, end, flank_extend=flank_extend)[1]
		# alpha records: a list formated as [[alpha, count], ...]
		alpha_records = [[rec[1].count('C')/(rec[1].count('C') + rec[1].count('T')), rec[-1]] for rec in pat_records if rec[1].count('C') + rec[1].count('T')>=4]
		if len(alpha_records) > 0:
			n_fragment = sum([rec[-1] for rec in alpha_records])
			return sum([rec[-1] for rec in alpha_records if (meth_direct=='hypo' and rec[0]<alpha_cut) or (meth_direct=='hyper' and rec[0]>alpha_cut)])/n_fragment, n_fragment
		else:
			return np.nan, 0

	def __str__(self):
		'''
		fetch basic information of this pat object, including:
		- pat file path
		- sample name
		- assem_version
		'''
		return f'pat_file: {self.__pat_file}\nsample_name: {self.__sample_name}\nassem_version: {self.__assem_version}'