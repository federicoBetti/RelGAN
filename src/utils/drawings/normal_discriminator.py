import sys
sys.path.append('../')
from pycore.tikzeng import *

# defined your arch
arch = [
    to_head( '..' ),
    to_cor(),
    to_begin(),
    to_Conv("conv1", 5000, 40, offset="(0,0,0)", to="(0,0,0)", height=2, depth=60, width=20 ),
    to_Conv("conv2", 512, 40, offset="(0,0,0)", to="(0,0,0)", height=2, depth=25, width=20, caption="embedding" ),
    to_ConvConvRelu("convconvrelu3", s_filer=256, n_filer=(64,64), offset="(0,0,0)", to="(0,0,0)", width=(2,2), height=40, depth=40, caption="conv + relu"),
    to_ConvConvRelu("convconvrelu3", s_filer=256, n_filer=(64,64), offset="(0,0,0)", to="(0,0,0)", width=(2,2), height=40, depth=40, caption="conv + relu"),
    to_Conv("conv2", 512, 40, offset="(0,0,0)", to="(0,0,0)", height=2, depth=25, width=20, caption="self attention" ),
    to_Pool("pool1", offset="(0,0,0)", to="(conv1-east)"),
    to_Conv("conv2", 128, 64, offset="(1,0,0)", to="(pool1-east)", height=32, depth=32, width=2 ),
    to_connection( "pool1", "conv2"),
    to_Pool("pool2", offset="(0,0,0)", to="(conv2-east)", height=28, depth=28, width=1),
    to_SoftMax("soft1", 10 ,"(3,0,0)", "(pool1-east)", caption="SOFT"  ),
    to_connection("pool2", "soft1"),
    to_end()
    ]

def main():
    namefile = str(sys.argv[0]).split('.')[0]
    to_generate(arch, namefile + '.tex' )

if __name__ == '__main__':
    main()