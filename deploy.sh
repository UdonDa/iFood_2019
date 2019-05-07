rsync -davzru --include-from <(git ls-files) --exclude .git --exclude-from <(git ls-files -o --directory) . im00:/home/yanai-lab/horita-d/conf/cvpr2020/ifood_challenge2019/
