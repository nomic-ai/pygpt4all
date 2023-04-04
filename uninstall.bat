@echo off
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

echo This will uninstall the environment. Are you sure? [Y/N] :
set /p choice=
if /i "%choice%" equ "Y" (
    REM Download Python installer
    echo -n 
    deactivate 2>nul || :    
    set /p="Removing virtual environment..." <nul
    powershell -Command "Remove-Item -Recurse -Force env"
    echo OK
    pause
) else (
    echo Please install Python and try again.
    pause
    exit /b 1
)
