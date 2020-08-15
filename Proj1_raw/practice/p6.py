from multiprocessing import Pool

def f(x):
	return x*x

if __name__ == '__main__':
    p = Pool(12)
    print(p.map(f, range(4)))