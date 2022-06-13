%--------------------------------------------------------------------------
%------- PLANTILLA DE CÓDIGO ----------------------------------------------
%------- Codigos generales ------------------------------------------------
%------- Por: David Fernández    david.fernandez@udea.edu.co --------------
%-------      Profesor Facultad de Ingenieria BLQ 21-409  -----------------
%-------      CC 71629489, Tel 2198528,  Wpp 3007106588 -------------------
%--------------------------------------------------------------------------

 % mirando lunares

%-------------------------------------------------------- dic 2021 --------
%--1. Inicializo el sistema -----------------------------------------------
%--------------------------------------------------------------------------

clear all   % Inicializa todas las variables
close all   % Cierra todas las ventanas, archivos y procesos abiertos
clc         % Limpia la ventana de comandosclc;

%--------------------------------------------------------------------------
%-- 2. Leemos las imágenes y realizamos la extracciòn  --------------------
%--------------------------------------------------------------------------

%---- a) Inicializa directorios


dir_source=(['.\Lunares\']);
%dir_target=(['.\Lunares_procesado\']);

% dos(['del ',dir_target]);
% mkdir(dir_target);
%---- b) Inventario de archivos a leer

files = dir([dir_source,'*.jpg']);
final=length(files);

%---- correr hasta que quede vacio


x=3;
for i=1:final-2
    [i,final]
    %archivo_1=[dir_source,files(i).name];
    a=imread([dir_source,files(i).name]);a=imresize(a,0.3);
    
    [fil,col,cap]=size(a);
    b=reshape(a,[fil,col*cap]);
    c = medfilt2(medfilt2(medfilt2(b,[x,x]),[x,x]),[x,x]);
    d=reshape(c,[fil,col,cap]);
    
    [y,s]=b_componentes_color_general(d); % y es en cmy ----------- s es en hsv
    
    figure(1); imshow([y, s]); title('Y(cmyk) and S (hsv)')
    
    % Tanto al Y(cmyk) como al S (hsv) se les aplica un graythresh

    y1=y;    l=graythresh(y1);    y1=im2bw(y1,l);    y1=uint8(y1)*255;

    s1=s;   l=graythresh(s1);   s1=im2bw(s1,l);     s1=uint8(s1)*255;

    figure(2);imshow([y1, s1]);    title("Y1 and S1 in BW after Threshold");
    
    % Rellenado de huecos, limpiado de bordes y area open para el Y e S
    y1=imfill(y1);
    
    y1=imclearborder(y1);
    y1=bwareaopen(y1,300);
    y1=uint8(y1)*255;

    figure(3);imshow([y1,y2,y3,y4]);    title("ImFill, ClearBorder, BWareaOpen, Uint*255");

    s1=imfill(s1);
    s=s1;

    s1=imclearborder(s1);
    c=s1;

    s1=bwareaopen(s1,300);
    b=s1;

    s1=uint8(s1)*255;
    figure(3);imshow([s,c,b]);title(["filled, clearborder, bwareaopen"]);
    
    y=cat(3,y,y,y);y1=cat(3,y1,y1,y1);
    s=cat(3,s,s,s);s1=cat(3,s1,s1,s1);
    
    figure(4);imshow([a,d,y;s,y1,s1]);
    title(['original resized yellow(cmyk)']);
    
    pause;
    close all;
    
end
 


