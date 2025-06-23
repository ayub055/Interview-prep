class Arrays:
    def remove_duplicates_sorted_array(self, nums):
        """
        Removes duplicates from a sorted array in-place and returns the new length.
        :param nums: List[int] - sorted array with possible duplicates
        :return: int - new length of the array without duplicates
        """
        if not nums:
            return 0
        
        write_index, n = 0, len(nums)
        for itr in range(n):
            if nums[write_index] != nums[itr]:
                write_index += 1
                nums[write_index] = nums[itr]
        
        return write_index + 1
    
if __name__ == "__main__":
    arr = Arrays()
    nums = [1]
    new_length = arr.remove_duplicates_sorted_array(nums)
    print(f"New length: {new_length}, Modified array: {nums[:new_length]}")