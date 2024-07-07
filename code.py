def numberOfSubarrays(nums, k):
    count = 0
    odd_count = 0
    prefix_counts = {0: 1}
    for num in nums:
        if num % 2 == 1:
            odd_count += 1
        if odd_count - k in prefix_counts:
            count += prefix_counts[odd_count - k]
        if odd_count in prefix_counts:
            prefix_counts[odd_count] += 1
        else:
            prefix_counts[odd_count] = 1
    return count
# 示例使用
nums = [2,2,2,1,2,2,1,2,2,2]
k = 2
print(numberOfSubarrays(nums, k))  # 輸出應該是 2