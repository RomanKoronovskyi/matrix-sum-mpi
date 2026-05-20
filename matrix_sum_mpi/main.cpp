#include <iostream>
#include <vector>
#include <mpi.h>
#include <ctime>
#include <cstdlib>
#include <iomanip>

using namespace std;

void PrintMatrix(const string& name, double* pMatrix, int M) {
    if (M > 10) return;
    cout << "Matrix " << name << ":" << endl;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < M; j++) {
            cout << setw(8) << fixed << setprecision(2) << pMatrix[i * M + j] << " ";
        }
        cout << endl;
    }
    cout << endl;
}

void DefineDistribution(int M, int ProcNum, vector<int>& sendCounts, vector<int>& displs) {
    int avgRows = M / ProcNum;
    int extraRows = M % ProcNum;
    int offset = 0;
    for (int i = 0; i < ProcNum; i++) {
        int rowsForProc = avgRows + (i < extraRows ? 1 : 0);
        sendCounts[i] = rowsForProc * M;
        displs[i] = offset;
        offset += sendCounts[i];
    }
}

void DataDistribution(double* pA, double* pB, double* pLocalA, double* pLocalB, int localSize,
    int* sendCounts, int* displs, int ProcRank) {
    MPI_Scatterv(pA, sendCounts, displs, MPI_DOUBLE, pLocalA, localSize, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(pB, sendCounts, displs, MPI_DOUBLE, pLocalB, localSize, MPI_DOUBLE, 0, MPI_COMM_WORLD);
}

void ParallelResultCalculation(double* pLocalA, double* pLocalB, double* pLocalC, int localSize) {
    for (int i = 0; i < localSize; i++) {
        pLocalC[i] = pLocalA[i] + pLocalB[i];
    }
}

void ResultCollection(double* pLocalC, double* pC, int localSize, int* sendCounts, int* displs) {
    MPI_Gatherv(pLocalC, localSize, MPI_DOUBLE, pC, sendCounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
}

int main(int argc, char* argv[]) {
    int ProcNum, ProcRank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &ProcNum);
    MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);

    int M = 100;

    double* pA = nullptr, * pB = nullptr, * pC = nullptr;
    vector<int> sendCounts(ProcNum);
    vector<int> displs(ProcNum);

    DefineDistribution(M, ProcNum, sendCounts, displs);

    if (ProcRank == 0) {
        pA = new double[M * M];
        pB = new double[M * M];
        pC = new double[M * M];

        srand(static_cast<unsigned int>(time(0)));
        for (int i = 0; i < M * M; i++) {
            pA[i] = static_cast<double>(rand() % 1000) / 100.0;
            pB[i] = static_cast<double>(rand() % 1000) / 100.0;
        }

        cout << "Matrix Size: " << M << "x" << M << ", Processes: " << ProcNum << endl;

        PrintMatrix("A", pA, M);
        PrintMatrix("B", pB, M);
    }

    double StartTime, EndTime;

    if (ProcNum == 1) {
        StartTime = MPI_Wtime();
        for (int i = 0; i < M * M; i++) {
            pC[i] = pA[i] + pB[i];
        }
        EndTime = MPI_Wtime();
        cout << "Mode: Serial" << endl;
    }

    else {
        int localSize = sendCounts[ProcRank];
        vector<double> localA(localSize), localB(localSize), localC(localSize);

        StartTime = MPI_Wtime();

        DataDistribution(pA, pB, localA.data(), localB.data(), localSize, sendCounts.data(), displs.data(), ProcRank);
        ParallelResultCalculation(localA.data(), localB.data(), localC.data(), localSize);
        ResultCollection(localC.data(), pC, localSize, sendCounts.data(), displs.data());

        EndTime = MPI_Wtime();
        if (ProcRank == 0) cout << "Mode: Parallel" << endl;
    }

    if (ProcRank == 0) {
        PrintMatrix("C (Result)", pC, M);

        cout << "Execution time: " << (EndTime - StartTime) << " sec." << endl;

        delete[] pA;
        delete[] pB;
        delete[] pC;
    }

    MPI_Finalize();
    return 0;
}