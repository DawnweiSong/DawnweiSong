#!/bin/bash

condarc=$HOME/bin/conda.bashrc

#eval "$('/c/bit9prog/dev/anaconda3/Scripts/conda.exe' 'shell.bash' 'hook')"
/c/bit9prog/dev/anaconda3/Scripts/conda.exe 'shell.bash' 'hook' >   $condarc

dos2unix $condarc 1>/dev/null 2>/dev/null

source $condarc

#fix path problem of anaconda3/Library/usr/bin
echo $PATH |/usr/bin/tr -s ':' '\n'  |/usr/bin/sed '/Library\/usr\/bin/d' > /tmp/path
/usr/bin/cat > /tmp/path2 <<eoc
$(/usr/bin/cat /tmp/path)
/c/bit9prog/dev/anaconda3/Library/usr/bin
eoc
export PATH=$( /usr/bin/cat /tmp/path2 | /usr/bin/tr -s '\n'  ':' )

conda info --envs

#remove the 1st \n of PS1 such that conda env does not force a newline 
export PS1="$(echo $PS1 |sed 's;\\n;;') "
conda activate tf27_py39
