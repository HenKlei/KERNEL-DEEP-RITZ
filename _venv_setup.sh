set -e
export BASEDIR="$(cd "$(dirname ${BASH_SOURCE[0]})" ;  pwd -P)"
cd "${BASEDIR}"

# Create and source virtualenv
if [ -e "${BASEDIR}/venv/bin/activate" ]; then
	echo "using existing virtualenv"
else	
	echo "creating virtualenv ..."
	virtualenv --python=python3 venv
fi

source venv/bin/activate

# Upgrade pip and install libraries
pip install --upgrade pip

pip install torch==2.1.1 torchvision==0.16.1 --index-url https://download.pytorch.org/whl/cpu
pip install git+https://gitlab.mathematik.uni-stuttgart.de/pub/ians-anm/2l-vkoga@v0.1.2
pip install tikzplotlib==0.10.1
