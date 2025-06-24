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
    
    def isAnagram(self, s, t):
        """
        Checks if two strings are anagrams of each other.
        :param s: str - first string
        :param t: str - second string
        :return: bool - True if s and t are anagrams, False otherwise
        """
        if len(s) != len(t):
            return False
        
        offset = ord('a')
        storage = [0] * 26
        for ch in s:
            ch_idx = ord(ch) - offset
            storage[ch_idx] += 1

        for ch in t:
            ch_idx = ord(ch) - offset
            if storage[ch_idx] == 0:
                return False
            
            storage[ch_idx] -= 1 

        for elem in storage:
            if elem != 0:
                return False
            
        return True
    
if __name__ == "__main__":
    arr = Arrays()
    nums = [1]
    new_length = arr.remove_duplicates_sorted_array(nums)
    print(f"New length: {new_length}, Modified array: {nums[:new_length]}")

    arr = Arrays()
    s = "jam"
    t = "jar"
    is_anagram = arr.isAnagram(s, t)
    print(f"Are '{s}' and '{t}' anagrams? {is_anagram}")