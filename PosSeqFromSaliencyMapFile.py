__author__ = 'jasper.zuallaert'
# Given a saliency map file (f), create two files with only positive samples:
# - One fasta file (.fasta), with all sequences (and their uniprot ids) of the positive samples
# - One saliency map file (.vis), with only positive samples from the initial file
def selectPosSeqFromFile(f):
    short_f = f[f.index('/')+1:f.index('.')]
    vis_lines = [line for line in open(f).readlines()]
    out_fasta_fname = 'seqFiles/short_f.fasta'
    out_vis_fname = 'seqFiles/short_f.vis'
    out_fasta = open(out_fasta_fname,'w')
    out_vis   = open(out_vis_fname,'w')

    ctr = 0
    for i in range(len(vis_lines)//3):
        labLine = vis_lines[i*3+0].strip()
        seqLine = vis_lines[i*3+1].strip()
        scrLine = vis_lines[i*3+2].strip()

        if labLine.split(',')[1] == '1':
            seq = seqLine.replace(',','')
            #import pdb
            #pdb.set_trace()
            print('>seq' + str(ctr), file=out_fasta)
            print(labLine + ' - seq' + str(ctr), file=out_vis)

            print(seq, file=out_fasta)
            print(seqLine, file=out_vis)

            print(scrLine, file=out_vis)

            ctr += 1
    return out_fasta_fname, out_vis_fname


