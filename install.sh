#!/usr/bin/bash

echo                            G#........................G                          
echo                         ............*....................                       
echo                   G*  *  ..*:#:::::G::::::::GG::::#:*...  .  .G                 
echo                G....  **  ::::::::GGG::::::GGG::::::::   **  ....:              
echo             G.......  ***   :::::GGGGG#:::GGGG:GG::::   **,  .......G           
echo           #..*...:::   ***,  :::::GGGGGGGGGGGGGG::::  ****  .::#......G         
echo         #......#:::::   .***  ::    .            ::  ,**.  .:::::.......G       
echo       G*.....:::::::::.   ,,                         *,   ::::::::::......:     
echo      :...**:::::::::::::. GGGGGGGGG.          GGGGGGGGG :::::::::::::#.*...G    
echo     ......#:::::::GGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGG:::::::#.....:   
echo    ......:::::::::GGSSGGSGGGGGGGGGGGGGG*  GGGGGGGGGGGGGSGGSSSGG:::::::::.....:  
echo   :.....::::::::::GGGGSGGG:GGG:GGGGGGGG   #GGGGGGS#GG#SSGGSGGGG::::::::::...*.G 
echo  #.....:::::::::::#GGGGGGGGGGGGGGS#GGG     GGG#GGGGGGGGGSGSGGG:::::::::::#*....:
echo  :....:::::::::::::GG#GSGGGGGGGGSSSSG        GSSGGGGGGGGGGGSG::::::::::::::....#
echo  *..*.::::::::::::::#GGGGGGGGGGGG,  ..........  ,GGGGGGGGGGG::::::::::::::#.....
echo :.*...:::::::::::::::::            GGGGG..GGGGG          :::::::::::::::::#.....
echo  *....#::::::::::::::::.      ,,,,,....GGGG....,,*,,     :::::::::::::::::#.....
echo  .....#:::::::::::::::::*       *G,,,,,,,,,,,,,S*,      ::::::::::::::::::#*...#
echo  G.....:::::::::::::::::::       .,,,*:::::.,,,,      *:::::::::::::::::::.....G
echo   ..**.G::::::::::::::::::::::   ...............   ::::::::::::::::::::::#..... 
echo   ......#::::::::::::::::::::::***.............***::::::::::::::::::::::#.....  
echo    #.....G:::::::::::::::::::::  .,,************  :::::::::::::::::::::#.....,  
echo     *.....*::::::::::::::::::::         .,.       ::::::::::::::::::::......    
echo       #.*...#::::::::::::::::::                   ::::::::::::::::::#.....G     
echo        #......:::::::::::::::::                   ::::::::::::::::#......*      
echo          G......G::::::::::::::                   :::::::::::::::......G        
echo            #..*....#:::::::::::                   :::::::::::G......*#          
echo               S........::::::::                   ::::::::........S             
echo                  S....*.....G::                   ::G..........S                
echo                      S.........                   .........S                    
echo                           G:.*.                   ...:G                                     
echo echo PYLLAMACPP Install tool
echo This will install the python bindings for llamacpp.

# Install Python 3.10 and pip
echo -n "Checking for python3.10..."
if command -v python3.10 > /dev/null 2>&1; then
  echo "OK"
else
  read -p "Python3.10 is not installed. Would you like to install Python3.10? [Y/N] " choice
  if [ "$choice" = "Y" ] || [ "$choice" = "y" ]; then
    echo "Installing Python3.10..."
    sudo apt update
    sudo apt install -y python3.10 python3.10-venv
  else
    echo "Please install Python3.10 and try again."
    exit 1
  fi
fi

# Install venv module
echo -n "Checking for venv module..."
if python3.10 -m venv env > /dev/null 2>&1; then
  echo "OK"
else
  read -p "venv module is not available. Would you like to install it? [Y/N] " choice
  if [ "$choice" = "Y" ] || [ "$choice" = "y" ]; then
    echo "Installing venv module..."
    sudo apt update
    sudo apt install -y python3.10-venv
  else
    echo "Please install venv module and try again."
    exit 1
  fi
fi

# Create a new virtual environment
echo -n "Creating virtual environment..."
python3.10 -m venv env
if [ $? -ne 0 ]; then
  echo "Failed to create virtual environment. Please check your Python installation and try again."
  exit 1
else
  echo "OK"
fi

# Activate the virtual environment
echo -n "Activating virtual environment..."
source env/bin/activate
echo "OK"

# Install the required packages
echo "Installing requirements..."
python3.10 -m pip install pip --upgrade
python3.10 -m pip install -r requirements.txt

if [ $? -ne 0 ]; then
  echo "Failed to install required packages. Please check your internet connection and try again."
  exit 1
fi

echo "Virtual environment created and packages installed successfully."


echo Installing pyLLamacpp
pip install -e .
exit 0
