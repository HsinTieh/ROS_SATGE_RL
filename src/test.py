# import yaml
# f = open(r'./config.yaml')
# y = yaml.load(f)
# print(y['age'])

# from mpi4py import MPI
# comm = MPI.COMM_WORLD
# size = comm.Get_size()
# rank = comm.Get_rank()
# data = (rank+1)**2
# data = comm.gather(data, root=0)
# if rank == 0:
#     print ("rank = %s " %rank + "...receiving data to other process")
#     for i in range(1, size):
#         data[i] = (i+1)**2
#         value = data[i]
#         print(" process %s receiving %s from process %s" % (rank , value , i))

# done =False
# while not done:
#     print(1)

# from mpi4py import MPI
# comm = MPI.COMM_WORLD
# rank = comm.Get_rank()
# size = comm.Get_size()
# if rank == 0:
#     array_to_share = [1, 2, 3, 4 ,5 ,6 ,7, 8 ,9 ]

# else:
#     array_to_share = None
# s =[]
# for i in range(size):
#     s.append(size)
# recvbuf = comm.scatter(s, root=0)
# print("process = %d" %rank + " recvbuf = %d " %recvbuf)


# import sac 



# with open("sac.py") as fp:
#     for i, line in enumerate(fp):
#         if "\xc9" in line:
#             print i, repr(line)
scan =None
while scan is None:
    print(1)
    pass
print(2)