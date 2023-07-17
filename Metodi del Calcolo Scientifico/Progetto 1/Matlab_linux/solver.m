clear;
clc;

% Loading matrices
matrices = dir("./matrices/*.mat");

file = ["Nome", "Dimensione", "Tempo(s)", "Memoria(Kb)", "Errore relativo", "NNZ", "Cond"];
writematrix(file, "report.csv", 'Delimiter', 'semi');

for matrix = matrices'
    name = convertCharsToStrings(matrix.name);
    matrix = load("./matrices/" + matrix.name);
    A = matrix.Problem.A;
    dim = size(A,1);

    %%% Solve linear system Ax=b
    % Exact solution
    xe = ones(size(A,1), 1);
    % Compute b
    b = A * xe;
    
    try
        profile clear;
        profile -memory on;
        
        tic
        x = solveSystem(A, b);
        time = toc;
        disp(time);
        
        profiler = profile('info');
        fNames = {profiler.FunctionTable.FunctionName};
        fRow = find(strcmp(fNames(:), 'solver>solveSystem'));
        mem = profiler.FunctionTable(fRow).TotalMemAllocated; 
        
        % Group info
        notZero = nnz(A);
        %cond_ = condest(A);
        err = norm(x-xe) / norm(xe);
        
        res = [name, dim, time, mem, err, notZero];
        
        writematrix(res, "report.csv", 'WriteMode', 'append', 'Delimiter', 'semi');
    catch exception
        disp(exception.message);
        res = [name dim "N/A" "N/A" "N/A" "N/A"];
    end
    % profile report;
end

function x = solveSystem(A, b)
    x = A \ b;
end