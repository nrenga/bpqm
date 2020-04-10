function A = kron_multi(M)

A = 1;
for j = 1:length(M)
    A = kron(A, M{j});
end

end