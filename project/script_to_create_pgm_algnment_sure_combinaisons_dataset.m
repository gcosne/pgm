%in order to evaluate precision, recall, and alignment error rate (AER) as 
%defined by Och and Ney (2003) we need to have the annotated set of sure 
% alignments S, P  the annotated set of possible alignments,
% and A denotes the set of alignments produced by the model under test

%here we create S from our small personal corpus

%pour S : je croise tous les outputs possibles de ibm1, 2 et hmm, 
%je ne garde que le max dans le tableau de résultats, je met tout le reste à 0

%pour P : je crois tous les outputs possibles de ibm1, 2 et hmm, 
%je répartit les poids uniformes entre valeurs non nulles

%pros de la méthode : plus rapide / robuste que de le faire à la main. Cons : overate positivement systématiquement ces 3 indices, comme S et P ont été construites à partir de A
%mais pour nous comme on a juste un mini-corpus & que l'on veut comparer de manière relative entre ibm1, 2 et hmm (et non de manière absolue avec "la vraie vie" c'est pour moi ce qui fait le plus de sens en le moins de temps)

%add relative path
%addpath('./murphy-hmm-master/HMM');

%load all output files
output_ibm1 = csvread('C:\Users\Nicolas\Documents\Cardabel\MVA\PGM_project\pgm\project\output_ibm1.csv',1,1);
output_ibm2 = csvread('C:\Users\Nicolas\Documents\Cardabel\MVA\PGM_project\pgm\project\output_ibm2.csv',1,1);
output_hmm = csvread('C:\Users\Nicolas\Documents\Cardabel\MVA\PGM_project\pgm\project\output_hmm.csv',1,1);

%create S & P database

%S using Max function for instance
S_output_table = zeros(size(output_hmm,1),size(output_hmm,2));
sum_matrix = (  output_hmm + output_ibm2 + output_ibm1 );
for i=1:size(output_hmm,2)
    [M,I] = max(sum_matrix(:,i));
    S_output_table(I,i) = 1;
end


%P using Uniform function on k largest elements only
for k = 2:25 %decide for k, k from 1 to size(output) which is equivalent to a uniform
%function
P_output_table = zeros(size(output_hmm,1),size(output_hmm,2));
for i=1:size(output_hmm,2)
    [sortedX,sortingIndices] = sort(sum_matrix(:,i),'descend');
    maxValues = sortedX(1:k);
    maxValueIndices = sortingIndices(1:k);
    P_output_table(maxValueIndices,i) = 1/k;
end

%% BEWARE : HERE I USED THE CSV FILES NOT FROM DECODING !!!! 
%calculate AER index
AER_index_output_ibm1(k) = 1 - (sum(sum(abs( output_ibm1 - P_output_table)+abs(output_ibm1 - S_output_table))))/(size(output_hmm,1) + size(output_hmm,2));
AER_index_output_ibm2(k) = 1 - (sum(sum(abs( output_ibm2 - P_output_table)+abs(output_ibm2 - S_output_table))))/(size(output_hmm,1) + size(output_hmm,2));
AER_index_output_hmm(k) = 1 - (sum(sum(abs( output_hmm - P_output_table)+abs(output_hmm - S_output_table))))/(size(output_ibm1,1) + size(output_ibm1,2));

end
AER_index_output_ibm1(1) = [];
AER_index_output_ibm2(1) = [];
AER_index_output_hmm(1) = [];
figure;

plot(AER_index_output_ibm1)
hold on
plot(AER_index_output_ibm2)
plot(AER_index_output_ibm2)
legend('IBM1', 'IBM2', 'HMM')
title('AER index for ibm1, ibm2 & hmm outputs with different methods for calculating the P datasets')
xlabel('k ie the number of possible alignments we fix for the P dataset')
ylabel('AER indices')
