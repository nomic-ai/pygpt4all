#!/bin/bash


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

echo "This will uninstall the environment. Are you sure? [Y/N]"
read choice
if [[ "$choice" =~ [yY] ]]; then
    # Download Python installer
    printf "Removing virtual environment..."
    rm -rf env
    echo "OK"
    read -p "Press [Enter] to continue..."
else
    echo "Please install Python and try again."
    read -p "Press [Enter] to continue..."
    exit 1
fi
