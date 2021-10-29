import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src/')))

from papyrus_scripts.utils import matchRCSB

class Startup(object):
    def __init__(self,wd,js_in,js_out,p_in,p_out):
        data = {
                 'wd'       : wd,
                 'js_in'    : js_in,
                 'js_out'   : js_out,
                 'p_in'     : p_in,
                 'p_out'    : p_out,
               }
        START = matchRCSB.Init(data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='matchRCSB',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description = '       == matchRCSB == ')
    
    parser.add_argument('--version', 
                        action='version', 
                        version='%(prog)s 0.1.0')

    parser.add_argument('-wd', '--workdir',
                        dest = "wd",
                        required = False,
                        help = "Optionally provide a new working directory")

        
    parser.add_argument('-js_in', '--json_infile',
                        dest = "js_in",
                        required = False,
                        default = None,
                        help = "Optionally specify a RCSB_data .json file")
        
    parser.add_argument('-js_out', '--json_outfile',
                        dest = "js_out",
                        required = False,
                        default = 'RCSB_data.json',
                        help = "Save the RCSB_data file under a different file name with this function")
        
    parser.add_argument('-p', '--papyrus_inputfile',
                        dest = "p_in",
                        required = True,
                        help = "Papyrus input datafile")
       
    parser.add_argument('-o', '--papyrus_outputfile',
                        dest = "p_out",
                        required = False,
                        default = 'Papyrus-RCSB_matched.tsv',
                        help = "Optionally provide a different filename for the RCSB match Papyrus entries")                    

    args = parser.parse_args()

    Startup(
            wd      = args.wd,
            js_in   = args.js_in,
            js_out  = args.js_out,
            p_in    = args.p_in,
            p_out   = args.p_out
           )        