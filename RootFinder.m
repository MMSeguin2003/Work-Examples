function [root, info] = RootFinder(func, Int, params)
    % Used to solve for a root of func in the interval Int
    % within a certain tolerance using a combination of
    % interquartile interpolation and bisection

    % Assigning all the variables from the inputs
    a = Int.a;
    b = Int.b;
    % Get parameters
    RTOL = params.root_tol;
    FTOL = params.func_tol;
    MAXIT = params.maxit;
    % Starting the iteration variables
    i = 1;
    FunCalls = 0;
    % Making the info output variable (0 is success)
    info.flag = 0;
    info.FunCalls = FunCalls;
    % Check inputs for validity
    if a >= b
        info.flag = 1;
        root = "Incorrect interval inputted, a should be less than b";
        return;
    elseif isfinite(a) == false || isfinite(b) == false
        info.flag = 1;
        root = "Incorrect interval inputted, a and b should be finite";
        return;
    elseif RTOL <= 0 || FTOL <= 0
        info.flag = 1;
        root = "Incorrect tolerance inputted, tolerance should be positive";
        return;
    elseif MAXIT <= 0
        info.flag = 1;
        root = "Incorrect max iterations inputted, max iterations should be positive";
        return;
    elseif sign(func(a))*sign(func(b)) > 0
        info.flag = 1;
        info.FunCalls = 2;
        root = "Interval may not contain a root";
        return;
    end
    FunCalls = FunCalls + 2;
    % Assigning the initial estimates as per the instructions
    x0 = a;
    x1 = b;
    x2 = (a+b)/2;
    % Starting loop and making sure we do not exceed maximum number of
    % function calls on this iteration
    while FunCalls + 4 <= MAXIT
        % Getting function values
        f0 = func(x0);
        f1 = func(x1);
        f2 = func(x2);
        FunCalls = FunCalls + 3;
        % Find the IQI
        x3 = x0*((f1*f2)/((f0 - f1)*(f0 - f2))) + x1*((f0*f2)/((f1 - f0)*(f1 - f2))) + x2*((f0*f1)/((f2 - f0)*(f2 - f1)));
        % Add the absolute function value of the estimate to the list
        AbsFuncVals(i) = abs(func(x3));
        FunCalls = FunCalls + 1;
        % If conditions are met stop the loop
        if (abs(x1 - x0) <= RTOL || AbsFuncVals(end) <= FTOL)
            info.FunCalls = FunCalls;
            root = x3;
            return;
        % First case to use bisection if IQI is out of interval as per the
        % instructions
        elseif (x3 < a || x3 > b)
            % Using bisection
            if sign(f0)*sign(f2) < 0
                a = x0;
                b = x2;
            else
                a = x2;
                b = x1;
            end
            x0 = a;
            x1 = b;
            x2 = (a+b)/2;
        % Second case to use bisection if IQI has not decreased by a factor
        % of 2 within 4 iterations as per the instructions but make sure
        % that we have already made at least 4 IQI estimates
        elseif (i > 4) & (AbsFuncVals(i) > 2*AbsFuncVals(i-4))
            % Using bisection
            if sign(f0)*sign(f2) < 0
                a = x0;
                b = x2;
            else
                a = x2;
                b = x1;
            end
            x0 = a;
            x1 = b;
            x2 = (a+b)/2;
        % Otherwise values have not been set for next iteration so
        % use IQI with safety bracket in instructions
        elseif sign(f0)*sign(f2) < 0
            x0 = x0;
            x1 = x2;
            x2 = x3;
        elseif sign(f1)*sign(f2) < 0
            x0 = x1;
            x1 = x2;
            x2 = x3;
        else
            break;
        end
        % Increasing iteration count
        i = i + 1;
    end
    % If we get to here the loop has ended without returning something so
    % we have gone past the max number of iterations.
    info.flag = 1;
    info.FunCalls = FunCalls;
    root = ["Root not found, last estimate is: " num2str(x3)];
end