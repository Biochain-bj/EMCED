#cython: language_level=3
import os, re, base_util, pyfaidx
import cython

def bam2pat(sam_file: str, pat_file: str, assem_version: str='hg19'):
	n_fragment: cython.int = 0
	n_empty: cython.int = 0    # no CpG site
	n_novalid: cython.int = 0  # no valid CpG site
	n_invalid: cython.int = 0  # mix valid and invalid CpG sites
	cigar_tuple: tuple[cython.int, cython.char]
	meth: list
	pos: list
	cur_read: list
	n: cython.int
	i: cython.int
	delect_shift: cython.int
	leftCorrect: cython.int
	cur_methStr: str
	indexs: list
	CpGindex: list
	cur_CpGindex: list
	pat: list
	bool_toinvalid: bool
	pos2index = base_util.COORDNATE_POS2INDEX(assem_version)
	seqDict = pyfaidx.Fasta(os.path.join(os.path.dirname(__file__), 'references', f'{assem_version}.fa'))
	with open(sam_file, 'r') as sam, open(pat_file, 'w') as out_pat, open(pat_file + '.invalid', 'w') as out_invalid:
		while True:
			paired_reads = [sam.readline().strip().split(), sam.readline().strip().split()]
			if not paired_reads[0] or not paired_reads[1]:
				break
			# deal with un-paired reads
			if paired_reads[0][0] != paired_reads[1][0]:
				while True:
					paired_reads = [paired_reads[1], sam.readline().strip().split()]
					if paired_reads[0][0] == paired_reads[1][0] or not paired_reads[0] or not paired_reads[1]:
						break
			if paired_reads[0][2].startswith('chr'):
				paired_reads[0][2] = paired_reads[1][2] = re.sub(r'^chr', '', paired_reads[0][2])
			n_fragment += 1
			meth = []
			pos = []
			# fetch overlap merged meth and pos for this paired-end reads
			for cur_read in paired_reads:
				# ['A00301:639:HH52JDSX5:2:1541:31575:16031_1:N:0:TGTATGCG+CCAGTTCA', '99', '10', '13650391', '42', '150M', '=', '13650482', '226', 'TTTGTTAGTTTGGGTTGTTTGATTGTTTTTTTTGTTGTTAAATTATTTTTTTGTTTTATTTTTTAGTATTGTTATTTTAGTTTTATAATATTAGTAGTTGTTTTTAGTTTTTTGTGTTTGTAATGATTATTTTTTTTTTGTTTGTTGTGT', 'FFFFFFFFFFFFFFFFFFFF:FFFFFFFFFFFFFFFFFFFFFFFFFFF::FFFF:FFFFFF:FFF,FFFFFFFFFFFFFFFFFFFFFFFFFFFFFF,::,FFFFFF:FF:FFF::FFFFFFFF,FFFFFFFFFFFFFFF,FFFF,F,,FF', 'NM:i:44', 'MD:Z:1C2C3C0C4C3C3C2C3C7C0C3C0C1C3C0C2C4C2C0C0C2C5C3C0C2C0C0C0C1C2C1C3C5C3C3C0C1C5C9C6C0C4C3C5', 'XM:Z:.x..h...hx....x...x...x..h...h.......hh...hh.h...hx..h....h..hhx..h.....h...hx..hhhh.h..h.h...x.....h...x...hh.x.....x.........h......hh....h...x.....', 'XR:Z:CT', 'XG:Z:CT']
				cur_read[3] = int(cur_read[3])
				# deal with read mapped with indel
				if re.search(r'I|D', cur_read[5]):
					n = 0
					delect_shift = 0
					cur_seqStr = ''
					cur_methStr = 'XM:Z:'
					for cigar_tuple in [(int(cur_read[5][s.start():s.end()]), cur_read[5][s.end()]) for s in re.finditer(r'\d+', cur_read[5])]:
						# delete for insert, and insert for delete
						if cigar_tuple[1] == 'M':
							cur_methStr += cur_read[-3][(n+5):(n+5+cigar_tuple[0]-delect_shift)]
							cur_seqStr += cur_read[9][(n):(n+cigar_tuple[0]-delect_shift)]
							n += cigar_tuple[0] - delect_shift
							delect_shift = 0
						elif cigar_tuple[1] == 'D':
							# deal with CpG site near delected site, switch with the 1st delected base, for example:
							# from:
							# CCCATCCCTTCAAACAATAAAAAACCCAACAAAACCCAC---AAAAAAAAAAAAAAAAAAAAATAAACAAAAACCCACACCCCCTATTATCTCCTCAACAACACCCCAAAACAAACACCCAATAATCACCCAACCCCCTACCCACAAACAACC
							# to
							# CCCATCCCTTCAAACAATAAAAAACCCAACAAAACCCACA---AAAAAAAAAAAAAAAAAAAATAAACAAAAACCCACACCCCCTATTATCTCCTCAACAACACCCCAAAACAAACACCCAATAATCACCCAACCCCCTACCCACAAACAACC
							# only switch for the case XG==GA
							if (cur_read[-3][n+5] == 'z' or cur_read[-3][n+5] == 'Z') and cur_read[-1][-2:] == 'GA':
								cur_methStr += cur_read[-3][n+5] + '-' * cigar_tuple[0]
								cur_seqStr += cur_read[9][n] + '-' * cigar_tuple[0]
								n += 1
								delect_shift = 1
							else:
								cur_methStr += '-' * cigar_tuple[0]
								cur_seqStr += '-' * cigar_tuple[0]
								delect_shift = 0
						elif cigar_tuple[1] == 'I':
							n += cigar_tuple[0] - delect_shift
							delect_shift = 0
					cur_read[-3] = cur_methStr
					cur_read[9] = cur_seqStr
				# left correction for OB or CTOB, recognize its strand
				leftCorrect = 1 if cur_read[-1][-2:] == 'GA' else 0
				# fetch meth and pos for this read, maybe need to merge with its previous mate read
				indexs = [s.start() for s in re.finditer('z|Z', cur_read[-3][5:])]
				if len(indexs) > 0:
					if len(pos) == 0:
						meth = [cur_read[-3][i+5] for i in indexs]
						pos = [i - leftCorrect + cur_read[3] for i in indexs]
					else:
						if indexs[0] - leftCorrect + cur_read[3] < pos[0]: # left append
							meth = [cur_read[-3][i+5] for i in indexs if i - leftCorrect + cur_read[3] < pos[0]] + meth
							pos = [i - leftCorrect + cur_read[3] for i in indexs if i - leftCorrect + cur_read[3] < pos[0]] + pos
						if indexs[-1] - leftCorrect + cur_read[3] > pos[-1]: # right append
							meth += [cur_read[-3][i+5] for i in indexs if i - leftCorrect + cur_read[3] > pos[-1]]
							pos += [i - leftCorrect + cur_read[3] for i in indexs if i - leftCorrect + cur_read[3] > pos[-1]]
			# generate pat string for this paired-end reads
			if len(pos) > 0:
				CpGindex = pos2index.convert_coordinate(paired_reads[0][2], pos[0], pos[-1], interchange='pos2index')
				bool_toinvalid = False
				if CpGindex[0] is not None:
					pat = ['.'] * (CpGindex[1] - CpGindex[0] + 1)
					for i in range(len(pos)):
						cur_CpGindex = pos2index.convert_coordinate(paired_reads[0][2], pos[i], pos[i], interchange='pos2index')
						if cur_CpGindex[0] is not None:
							pat[cur_CpGindex[0] - CpGindex[0]] = 'C' if meth[i] == 'Z' else 'T'
					if pat.count('C') + pat.count('T') > 0:
						out_pat.write('\t'.join([paired_reads[0][2], str(CpGindex[0]), ''.join(pat)]) + '\n')
						if pat.count('C') + pat.count('T') < len(pos):
							out_invalid.write(f'# exist invalid CpG:\n{paired_reads[0]}\n{paired_reads[1]}\n')
							n_invalid += 1
							bool_toinvalid = True
					else:
						out_invalid.write(f'# no valid CpG:\n{paired_reads[0]}\n{paired_reads[1]}\n')
						n_novalid += 1
						bool_toinvalid = True
				else:
					out_invalid.write(f'# no valid CpG:\n{paired_reads[0]}\n{paired_reads[1]}\n')
					n_novalid += 1
					bool_toinvalid = True
				if bool_toinvalid:
					for cur_read in paired_reads: out_invalid.write(f'{seqDict[paired_reads[0][2]][(cur_read[3] - 1):(cur_read[3] - 1 + len(cur_read[-3]) - 5)].seq}\n{seqDict[paired_reads[0][2]][(cur_read[3] - 1):(cur_read[3] - 1 + len(cur_read[-3]) - 5)].seq.replace(list(cur_read[-1][-2:])[0], list(cur_read[-1][-2:])[1])}\n{cur_read[9]}\n{cur_read[-3][5:]}\n')
			else:
				n_empty += 1
	return [n_fragment, n_empty, n_novalid, n_invalid]