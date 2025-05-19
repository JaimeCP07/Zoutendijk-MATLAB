clear
format long

%El programa minimiza una función sujeto a una serie de restricciones

%Este programa hace uso de la librería simbólica de Matlab, Se barajó la
%idea de eleborarlo sin su uso pero requería ,por parte del usuario, del
%calculo de los respectivos gradiantes de las restricciones y la función,
%asi que es pos de la comodidad, se descartó

%Para facilitar el introducir los datos del problema, se han incluido en un
%archivo aparte, al cual llamamos ahora:

%El problema usado como ejemplo se corresponde al ejercicio 1 de la hoja 3,apartado c)

% Los simblos ######### representan datos que han de ser proporcionados por el usuario del programa

x0 = [-2, -5.2];       %Punto desde donde arrancar el método, ha de ser factible (notación [x,y] )            ###########
Nvar = 2;         %indicamos el número de variables                                       ###########
Nres = 3;         %indicamos el número de restricciones del problema                      ###########
tolerancia_salida = 1e-5; %tolerancia de salida del método, se puede modificar a gusto del usuario   ###########
plot = 1;        %si se quiere graficar la función, poner 1, si no, 0.                      ###########
iter_info = 0; %Mostrar información de cada iteración                      ###########
%--------------------------------------------------------------------

syms x [1 Nvar]       %creamos el elemento simbólico vector de variables
syms g [1 Nres]       %creamos el elemento simbolico vector de restricciones 

% además, las restricciones serán introducidas ya corregidas para quedar de la forma <= 0
% Introducir en la f la funcion a MINIMIZAR (cambiar de signo si quiere maximizarse), y en las g(i) las restricciones

f    = symfun(     cos(x(1)) + sin(x(2))     ,x);     %La x tras la coma indica qué son las variables            ###########

g(1) = symfun(     x(1)^2 + 2*x(2) - 7            ,x);     %El usuario solo ha de modificar las funciones             ###########
g(2) = symfun(     x(1) + x(2) - 4/3      ,x);     %Las variables se introduciran como                        ###########
g(3) = symfun(     x(1)^3 - exp(-(x(2))) - 2                  ,x);     %elementos de un vector x, es decir, de la forma x(i)      ########### 

f_lambda =  @(lambda) (xk(0) + lambda * d(0))*(xk(1) + lambda * d(1));
if plot == 1
    
    % Plot using fsurf
    fsurf(f, [-10, 10, -10, 10], 'ShowContours','on')
    xlabel('x'), ylabel('y'), zlabel('f(x, y)')
    hold on

    % Dibujamos el punto inicial
    plot3(x0(1), x0(2), subs(f, x, x0), 'kx', 'MarkerSize', 10, 'LineWidth', 2)
    trajectory(:,1) = x0;

end

%--------------------------------------------------------------------

%Confirmamos que el x0 dado es factible

ValRes0 = subs(g, x, x0); %Evaluacion de las g en x0
for i = 1:Nres 
    if ValRes0(i) > 0
        error('El punto x0 no es factible')
    end
end

%Ahora calculamos los gradientes de la función y las restricciones

syms Df [1 Nvar] %creamos el elemento simbólico que almacena las derivadas parciales de la función
for i= 1:Nvar
    Df(i) = diff(f,x(i));    %Y le damos el valor de cada parcial
end

syms Dg [Nres Nvar] %creamos una matriz simbólica donde cada fila almacena gradiente de una restricción
for i = 1:Nres
    for j = 1:Nvar
        Dg(i,j) = diff(g(i),x(j));       %Y le damos el valor por restricción y variable
    end
end

%Ahora sí, iniciamos el método

k=1;
xk=x0;
z_old = 0; 

while true  %el bucle se repetirá hasta que el método encuentre el optimo
    if iter_info == 1
        disp('-------------------------------------------------------')
        disp('Iteración:')
        disp(k)
        disp('Punto actual:')
        disp(xk) %mostramos el punto actual
        value = subs(f, x, xk); %evaluamos la función en el punto xk
        disp('El valor de la función es:')
        disp(double(value))
    end
 
    %localizamos las restricciones activas

    ValRes = subs(g, x, x0);
    I = zeros(1,Nres); %vector con un 1 si la restriccion es activa, 0 si no lo es
    for i = 1:Nres
        if ValRes(i) == 0
            I(i) = 1;
        end
    end
        
    N_ResAct = sum(I); %Número total de restricciones activas

    %--------------SIMPLEX----------------

    % Ahora, minimicemos el problema de programación lineal aproximado usando el método del simplex  

    %En nuestro uso concreto tenemos que tanto z como las d(i) no
    %tienen porque ser positivas, luego hemos dedesdoblarlas en z= z+ - z-, 
    % d(i)= d(i)+ - d(i)-, y al tomar la solucion, interpretar correctamente el resultado

    c = [1 -1 zeros(1,2*Nvar)]; %ordenados: z+, z-, d1+, d1-,...
    b = [zeros(1,1+N_ResAct) ones(1,2*Nvar)]; %el del gradiente de f, los de gi y los de las restricciones de las d (di+ - di- <= 1)

    %definamos las correspontientes restricciones de nuestro simplex auxiliar

    Df_xk = subs(Df,x,xk); %evaluación del gradiente de f en xk
    Dg_xk = subs(Dg,x,xk); %evaluación del gradiente de gi en xk

    %Para ver el proceso mas claro, defianmos las restricciones por secciones y concatenemoslas
    %Recordemos ordenados: z+, z-, d1+, d1-,...

    %gradiente de f
    V_aux = [];
    for i = 1:Nvar
    V_aux = [V_aux Df_xk(i) -Df_xk(i)];
    end
    Res_Gradf = [-1 1 V_aux];

    %gradiente de gi
    M_aux = [];
    for j = 1:Nres
        if I(j)==1          %solo si la restricción es activa
            V_aux = [];
            for i = 1:Nvar
                V_aux = [V_aux Dg_xk(j,i) -Dg_xk(j,i)];
            end
            M_aux = [M_aux;[-1 1 V_aux]];
        end
    end
    Res_Gradgi = M_aux;

    % restricciones dj pertenece [-1,1]
    M_aux=[];
    for j = 1:Nvar
        V_aux = [zeros(1,2*j) 1 -1 zeros(1,2*(Nvar-j))];
        M_aux = [M_aux; V_aux];
        V_aux = [zeros(1,2*j) -1 1 zeros(1,2*(Nvar-j))];    
        M_aux = [M_aux; V_aux];
    end
    Res_dj = M_aux;

    %con esto tenemos nuestra matriz A
    A = [Res_Gradf;Res_Gradgi;Res_dj];

    [d_doble,z]= simplex_min(A, b, c); %La solución del problema lineal

    %Como las variables estaban desdobladas, recuperemoslas

    d=zeros(1,(length(d_doble)-2)/2); %d es la direccion factible de descenso
    for i=1:length(d)
        d(i)=d_doble(2*i+1)-d_doble(2*i+2);
    end
    if iter_info == 1 
        disp('d es:')
        disp(double(d))
        disp('z es:')
        disp(double(z))
    end



    %---------FIN DE SIMPLEX----------

    %Al resolver con el simplex el problema anterior linealizado hemos obtenido
    %el valor de z y el vector de desdenso optimo d

    if abs(z - z_old) <= tolerancia_salida
        disp('El óptimo es:')
        disp(xk)
        disp('El valor de la función es:')
        value = subs(f, x, xk); %evaluamos la función en el punto xk
        disp(double(value))
        if plot == 1
            %Ponemos el punto óptimo en la gráfica
            plot3(xk(1), xk(2), subs(f, x, xk), 'ypentagram', 'MarkerSize', 10, 'LineWidth', 2)
            z_vals = arrayfun(@(i) f(trajectory(1,i), trajectory(2,i)), 1:size(trajectory,2));
            plot3(trajectory(1,:), trajectory(2,:), z_vals, 'w-', 'LineWidth', 5)   % Path line
            hold off
        end
        return
    end

   

    if abs(z - z_old) > tolerancia_salida
    %Optimizamos en 1 variable lambda

        lambda_M = 0;
        paso = 0.1;
        tol = 1e-5; %tolerancia para la busqueda binaria
        flag_not_comp = true;
        iter_lim = 0;
        max_iter = 1e6; %número máximo de iteraciones para la busqueda binaria

        %Aproximamos el lambda_M
        %lambda_M es el máximo valor que puede tomar lambda, para que xk+lambda*d siga siendo factible
        %Para ello vamos a usar una busqueda binaria
        %Buscamos un punto que no está en la región factible en la dirección de descenso
        while flag_not_comp
            lambda_M = lambda_M + paso; %aumentamos lambda_M para buscar un punto que no cumpla las restricciones
            ValRes = subs(g, x, xk+lambda_M*d); %evaluamos las restricciones en el punto x0+lambda_M*d
            for i = 1:Nres
                if ValRes(i) > 0 %si alguna restricción no se cumple, dejamos de buscar
                    flag_not_comp = flag_not_comp&&false;
                end
            end
            iter_lim = iter_lim + 1; %aumentamos el número de iteraciones
            if iter_lim > max_iter
                disp('Se han hecho las siguientes iteraciones sin encontrar un punto que no cumpla las restricciones')
                disp(iter_lim)
                error('No se ha encontrado un punto que no cumpla las restricciones')
            end
        end




        left = 0;
        right = lambda_M;
        all_rest = true;
        while right - left > tol %mientras la diferencia entre los dos puntos sea mayor que la tolerancia
            lambda_M = (left+right)/2; %buscamos el punto medio
            point = xk+lambda_M*d; %evaluamos el punto medio
            ValRes = subs(g, x, point); %evaluamos las restricciones en el punto x0+lambda*d
            for i = 1:Nres
                if ValRes(i) > 0 %si alguna restricción no se cumple, seguimos buscando
                    right = lambda_M;
                    all_rest = false;
                    %disp('Valor de right cambiado')
                end
            end
            if all_rest
                left = lambda_M;
                %disp('Valor de left cambiado')
            end
            all_rest = true;
            % %disp('Ahora mismo right-left es')
            % %disp(right - left)
        end
        


        all_rest = true;
        ValRes = subs(g, x, point);
         %Nos aseguramos que el punto medio cumple las restricciones
         for i = 1:Nres
            if ValRes(i) > 0 %si alguna restricción no se cumple, seguimos buscando
                all_rest = false;
            end
        end
        if all_rest
            lambda_M = right;
        else
            lambda_M = left;
        end

        point = xk+lambda_M*d;
        if iter_info == 1
            disp('lambda_M es:')
            disp(double(lambda_M)) %mostramos el valor de lambda_M

            disp('El punto es:')
            disp(double(point)) %mostramos el punto medio
        end


syms lambda           % definimos lambda simbólico
x_lambda = xk + lambda * d;    % esto es un vector fila simbólico 1x2
f_lambda_sym = subs(f, x, x_lambda);  % sustituimos x por xk + lambda*d
f_lambda_num = matlabFunction(f_lambda_sym);
fun = @(lambda) f_lambda_num(0,0,lambda);
[lambda_k, fval] = fminbnd(fun, 0, lambda_M);
        
        xk=xk+lambda_k*d;
        k=k+1;
        z_old = z;
    end
    if plot == 1
        % Ponemos el punto en la gráfica
        plot3(xk(1), xk(2), subs(f, x, xk), 'bo', 'MarkerSize', 5, 'LineWidth', 2)
        pause(0.2)    % Pause to visualize step
        trajectory(:,k+1) = xk;
    end
end


%---------------CODIGO DEL SIMPLEX--------------------

function [Sol_optima, Val_optimo] = simplex_min(A, b, c)
% Método Simplex para Minimización con restricciones de = ó <= y las variables >= 0
% A -> matriz de restricciones (coeficientes)
% b -> vector de términos independientes
% c -> vector de coeficientes de la función objetivo (minimización)(incluye los costes 0)
% Devuelve la solución óptima y el valor óptimo

% Número de restricciones y variables
[m, n] = size(A);

% Construcción de la tabla simplex con variables de holgura
tablasimplex = [A eye(m) b'; c zeros(1, m+1)];

% A los coeficientes de las variables en las restricciones, eye(m) las variables de holgura, b las constantes de las restricciones
% En la parte inferior de la tabla, los costes de la función objetivo añadiendo los costes cero de las variables de holgura

% Variables básicas iniciales (las variables de holgura)
var_basicas = n+1:n+m;

% Iteración del método simplex
    while any(tablasimplex(end, 1:end-1) < 0) % Mientras haya costos negativos
        % Encontrar la columna pivote (variable entrante)
        [~, var_entrante] = min(tablasimplex(end, 1:end-1));

        % Criterio de razón mínima (para encontrar la fila pivote)
        ratios = tablasimplex(1:end-1, end) ./ tablasimplex(1:end-1, var_entrante);
        ratios(tablasimplex(1:end-1, var_entrante) <= 0) = Inf; % Evitar valores negativos o cero

        [~, fila_pivote] = min(ratios);

        if isinf(ratios(fila_pivote))
        error('El problema es no acotado.');
        end

        % Pivote de elementos
        elemento_pivote = tablasimplex(fila_pivote, var_entrante);
        tablasimplex(fila_pivote, :) = tablasimplex(fila_pivote, :) / elemento_pivote;

        for i = [1:fila_pivote-1, fila_pivote+1:m+1]
            tablasimplex(i, :) = tablasimplex(i, :) - tablasimplex(i, var_entrante) * tablasimplex(fila_pivote, :);
        end

        % Actualizar variables básicas
        var_basicas(fila_pivote) = var_entrante;
    end

% Extraer la solución óptima
Sol_optima = zeros(n, 1);
    for i = 1:m
        if var_basicas(i) <= n
            Sol_optima(var_basicas(i)) = tablasimplex(i, end);
        end 
    end

% Valor óptimo de la función objetivo 
% (el signo negativo viene de una adaptación de la maximización)
Val_optimo = -tablasimplex(end, end);
end
