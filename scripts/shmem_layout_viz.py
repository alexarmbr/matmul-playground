from enum import Enum
from copy import deepcopy

class dtype(Enum):
  INT = 1
  FLOAT4 = 2

class ArrayElement:
  """
  a class used to represent a packed array element for visualization purposes
  """

  def __init__(self, row : int, col : range):
    self.row = row
    self.col = col
  
  def __str__(self):
    return f"A[{self.row}, {self.col.start}..{self.col.stop-1}]"
  
  def __len__(self):
    return len(str(self))

class Array:
  """
  A class used to represent a 2D array of elements for visualization purposes
  """
  def __init__(self, rows : int, cols : int, dtype : dtype):
    if dtype == dtype.INT:
      # if we reinterpret cast an array of halfs to an array of ints, each addressable int contains two halfs packed together
      self.packed_elements = 2
    elif dtype == dtype.FLOAT4:
      # if we reinterpret cast an array of halfs to an array of float4s, each addressable float4 contains eight halfs packed together
      self.packed_elements = 8
    self.banks_per_packed_element = int(self.packed_elements / 2)
    self.rows = rows
    self.cols = int(cols / self.packed_elements)
    self.dtype = dtype
    self.array = [[ArrayElement(i, range(j,j + self.packed_elements)) for j in range(0, self.cols * self.packed_elements, self.packed_elements)] for i in range(self.rows)]
    
    # construct corresponding array with the same number of rows and columns that tracks which memory banks store each element
    self.memory_bank_layout = []
    bank_index = 0
    for i in range(self.rows):
      row = []
      for j in range(self.cols):
        row.append(range(bank_index, bank_index + self.banks_per_packed_element))
        bank_index = (bank_index + self.banks_per_packed_element) % 32
      self.memory_bank_layout.append(row)
  
  def count_bank_conflicts(self) -> int:
    """
    count the number of memory bank conflicts that will occur upon accessing any given column of array elements

    do this by counting how many memory banks store the 0th column of elements
    """
    
    # count how many memory banks store the 0th column of elements
    col_0_memory_banks = set()
    col_0_elements = self.array[0][0].col
    # import pdb; pdb.set_trace()
    for i in range(self.rows):
      for j in range(self.cols):
        if self.array[i][j].col == col_0_elements:
          col_0_memory_banks = col_0_memory_banks.union(list(self.memory_bank_layout[i][j]))
    print("memory banks storing 0th column: ", col_0_memory_banks)

    # calculate actual wavefronts, ideal wavefronts, and number of bank conflicts
    bytes_per_column = self.rows * self.packed_elements * 2 # elements per column * halfs per element * bytes per half
    bytes_per_ideal_wavefront = 32 * 4 # 32 memory banks * four bytes per bank
    bytes_per_actual_wavefront = len(col_0_memory_banks) * 4 # how many banks are storing the 0th column * four bytes per bank

    ideal_wavefronts = bytes_per_column / bytes_per_ideal_wavefront
    actual_wavefronts = bytes_per_column / bytes_per_actual_wavefront
    bank_conflicts = actual_wavefronts / ideal_wavefronts
    print("ideal wavefronts: ", ideal_wavefronts)
    print("actual wavefronts: ", actual_wavefronts)
    if bank_conflicts == 1:
      print("no bank conflicts")
    else:
      print(f"{bank_conflicts} way bank conflict")

  def __str__(self) -> str:
    """
    pretty print the array with columns aligned
    """
    column_lengths = [max(len(self.array[row][col]) for row in range(self.rows)) for col in range(self.cols)]
    string = ""
    for row in range(self.rows):
      for col in range(self.cols):
        string += str(self.array[row][col]).ljust(column_lengths[col]) + "  "
      string += "\n"
    return string
  
  def swizzle_array(self, swizzle_func : callable) -> 'Array':
    """
    apply swizzle function to array

    Args:
        swizzle_func (callable): a callable that takes in a row and column index and returns a tuple of the swizzled row and column index

    Returns:
        Array: a new Array object with the elements permuted according to the swizzle function
    """
    swizzled_array = deepcopy(self)
    for row in range(self.rows):
      for col in range(self.cols):
        swizzled_row, swizzled_col = swizzle_func(row, col, self.cols)
        swizzled_array.array[swizzled_row][swizzled_col] = self.array[row][col]
    return swizzled_array

def SWIZZLE_A(row, col, STRIDE):
  """
  swizzle function applied to A in shared memory
  swizzle is computed WRT to indices of the packed/vectorized elements
  i.e. float4/int rather than half
  """
  i = row * STRIDE + col
  swizzled_i = i ^ ((i & 0b10000) >> 4)
  swizzled_i = swizzled_i ^ ((swizzled_i & 0b1100) >> 2)
  swizzled_row = swizzled_i // STRIDE
  swizzled_col = swizzled_i % STRIDE
  return swizzled_row, swizzled_col


def SWIZZLE_C(row, col, STRIDE):
  """
  swizzle function applied to C in shared memory
  swizzle is computed WRT to indices of the packed/vectorized elements
  i.e. float4/int rather than half
  """
  i = row * STRIDE + col
  swizzled_i = i ^ ((i & 0b1110000) >> 3)
  swizzled_row = swizzled_i // STRIDE
  swizzled_col = swizzled_i % STRIDE
  return swizzled_row, swizzled_col

def SWIZZLE_B(row, col, STRIDE):
  i = row * STRIDE + col
  swizzled_i = i ^ ((i & 0b11100000) >> 5)
  swizzled_row = swizzled_i // STRIDE
  swizzled_col = swizzled_i % STRIDE
  return swizzled_row, swizzled_col


if __name__ == "__main__":

  # block tile dimensions for kernel 9
  # BM = 256
  BN = 256
  BK = 8
  
  # pattern of swizzled elements repeats every 8 rows, so we only need to look at the first 8 rows
  # ROWS = 8
  B_shmem = Array(BK, BN, dtype.FLOAT4)
  print(B_shmem)
  B_shmem_swizzled = B_shmem.swizzle_array(SWIZZLE_B)
  print("SWIZZLED: ")
  print(B_shmem_swizzled)

  stride = B_shmem.cols
  indices = [i * stride for i in range(8)]
  indices = [i ^ ((i & 0b11100000) >> 5) for i in indices]
  xor_patterns = [0b10, 0b110, 0b10, 0b110]


  for j in range(4):
    for i in indices:
      s_row = i // stride
      s_col = i % stride
      print(f"swizzled: {B_shmem_swizzled.array[s_row][s_col]}")
    print("--" * 10)
    indices = [i ^ xor_patterns[j] for i in indices]




