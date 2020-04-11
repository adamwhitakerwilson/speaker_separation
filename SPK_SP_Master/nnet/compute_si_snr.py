#!/usr/bin/env python

# wujian@2018 modified by Adam Whitaker-Wilson

"""
Compute SI-SDR as the evaluation metric
"""
from scipy.spatial.distance import euclidean, mahalanobis
from scipy.spatial.distance import cosine, sqeuclidean
import numpy as np
from fastdtw import fastdtw
import time
import wave, os, glob, librosa
from sklearn.metrics.pairwise import cosine_similarity
import argparse
from tqdm import tqdm
from collections import defaultdict
from libs.metric import si_snr, permute_si_snr
from libs.audio import WaveReader, Reader


def truncate(f, n):
    '''Truncates/pads a float f to n decimal places without rounding'''
    s = '{}'.format(f)
    if 'e' in s or 'E' in s:
        return '{0:.{1}f}'.format(f, n)
    i, p, d = s.partition('.')
    return '.'.join([i, (d+'0'*n)[:n]])

# https://github.com/kaituoxu/Conv-TasNet/blob/master/src/evaluate.py
def cal_SISNR(ref_sig, out_sig, eps=1e-8):
    """Calcuate Scale-Invariant Source-to-Noise Ratio (SI-SNR)
    Args:
        ref_sig: numpy.ndarray, [T]
        out_sig: numpy.ndarray, [T]
    Returns:
        SISNR
    """
    assert len(ref_sig) == len(out_sig)
    ref_sig = ref_sig - np.mean(ref_sig)
    out_sig = out_sig - np.mean(out_sig)
    ref_energy = np.sum(ref_sig ** 2) + eps
    proj = np.sum(ref_sig * out_sig) * ref_sig / ref_energy
    noise = out_sig - proj
    ratio = np.sum(proj ** 2) / (np.sum(noise ** 2) + eps)
    sisnr = 10 * np.log(ratio + eps) / np.log(10.0)
    return sisnr


# https://skipperkongen.dk/2018/09/19/cosine-similarity-in-python/
'''' Calulate cosine-similarity, dynamic time warp and si-snr between 2 vectors'''
def cos_sim(a,b, patha, pathb, f):

    
    # validate vector shapes
    if not (a.shape == b.shape):
        temp = np.ones(a.shape)
        temp2 = np.zeros(a.shape)
        np.place(temp2,temp,b)
        b = temp2    

    dot = np.dot(a, b)
    norma = np.linalg.norm(a)
    normb = np.linalg.norm(b)
    cos = dot / (norma * normb)
    
    # use library, operates on sets of vectors
    aa = a.reshape(1,len(a.tolist()))
    ba = b.reshape(1,len(b.tolist()))

    # manually compute cosine similarity, dynamic time warp and si-snr
    cos_lib = cosine_similarity(aa, ba)
    snr = cal_SISNR(b,a)
    dtw_val, path = fastdtw(a,b, dist=sqeuclidean)

    # write results to a file
    f.write(str(patha)+","+ str(pathb)+","+ str(dot)+","+ str(norma)+","+ str(normb)+","+ str(cos)+","+ str(cos_lib[0][0])+","+str(dtw_val)+","+str(snr)+"\n")

class SpeakersReader(object):
    def __init__(self, scps):
        split_scps = scps.split(",")
        if len(split_scps) == 1:
            raise RuntimeError(
                "Construct SpeakersReader need more than one script, got {}".
                format(scps))
        self.readers = [WaveReader(scp) for scp in split_scps]

    def __len__(self):
        first_reader = self.readers[0]
        return len(first_reader)

    def __getitem__(self, key):
        return [reader[key] for reader in self.readers]

    def __iter__(self):
        first_reader = self.readers[0]
        for key in first_reader.index_keys:
            yield key, self[key]


class Report(object):
    def __init__(self, spk2gender=None):
        self.s2g = Reader(spk2gender) if spk2gender else None
        self.snr = defaultdict(float)
        self.cnt = defaultdict(int)
        self.snr_ = []
        self.cnt_ = []

    def add(self, key, val):
        gender = 0
        if self.s2g:
            gender = self.s2g[key]
        self.snr[gender] += val
        self.cnt[gender] += 1
        self.snr_.append(val)
        self.cnt_.append(1)

    def report(self):
        
        print("SI-SDR(dB) Report: ")
        for gender in self.snr:
            tot_snrs = self.snr[gender]
            num_utts = self.cnt.get(0)

            print("{}: {:d}/{:.3f}".format(gender, num_utts,tot_snrs / num_utts))

def run(args, f):
    
    single_speaker = len(args.sep_scp.split(",")) == 1
    reporter = Report(args.spk2gender)
    
    
    if single_speaker:
        sep_reader = WaveReader(args.sep_scp)
        ref_reader = WaveReader(args.ref_scp)
        for key, sep in tqdm(sep_reader):
            ref = ref_reader[key]
            if sep.size != ref.size:
                end = min(sep.size, ref.size)
                sep = sep[:end]
                ref = ref[:end]
            snr = si_snr(sep, ref)
            reporter.add(key, snr)
    else:
        sep_reader = SpeakersReader(args.sep_scp)
        ref_reader = SpeakersReader(args.ref_scp)
        for key, sep_list in tqdm(sep_reader):
            ref_list_ = ref_reader[key]
            for i in range(len(sep_list)):
                for j in range(len(sep_list)):
                    cos_sim(sep_list[i], ref_list_[j], "s"+str(i), "r"+str(j), f)
    reporter.report()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=
        "Command to compute SI-SDR, as metric of the separation quality",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "sep_scp",
        type=str,
        help="Separated speech scripts, waiting for measure"
        "(support multi-speaker, egs: spk1.scp,spk2.scp)")
    parser.add_argument(
        "ref_scp",
        type=str,
        help="Reference speech scripts, as ground truth for"
        " SI-SDR computation")
    parser.add_argument(
        "--spk2gender",
        type=str,
        default="",
        help="If assigned, report results per gender")
    args = parser.parse_args()

    # timestamp for results file
    t = int(time.time())
    tup_str = tuple(args.sep_scp)
    tup = str(tup_str[11:20])

    # create results csv
    f = open("cos_sim_results"+"_"+tup+"_"+str(t)+".csv", "w")
    run(args, f)
    f.close()