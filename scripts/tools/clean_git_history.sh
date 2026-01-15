#!/bin/bash
# Git 历史清理脚本 - 移除敏感数据

cd /home/human/stock-underdog-ml

echo "⚠️  警告：此操作将重写 Git 历史，不可逆！"
echo "请确保：1) 已备份重要数据  2) 通知协作者"
echo ""
read -p "确定要继续吗？(输入 YES 继续): " confirm

if [ "$confirm" != "YES" ]; then
    echo "❌ 操作已取消"
    exit 1
fi

echo ""
echo "📋 步骤 1/4: 创建敏感数据替换文件..."

cat > /tmp/credentials-to-remove.txt << 'EOF'
# 示例格式：
# old_secret==>new_placeholder
# 
# 请在此处添加需要清理的敏感数据
# 例如：
# my_password123==>your_password_here
# api_key_xyz==>your_api_key_here
EOF

echo "✅ 替换文件已创建"
echo ""
echo "📋 步骤 2/4: 安装 BFG Repo-Cleaner..."

if [ ! -f "bfg-1.14.0.jar" ]; then
    wget -q https://repo1.maven.org/maven2/com/madgag/bfg/1.14.0/bfg-1.14.0.jar
    echo "✅ BFG 下载完成"
else
    echo "✅ BFG 已存在"
fi

echo ""
echo "📋 步骤 3/4: 清理 Git 历史..."
java -jar bfg-1.14.0.jar --replace-text /tmp/credentials-to-remove.txt .git

echo ""
echo "📋 步骤 4/4: 清理引用和垃圾回收..."
git reflog expire --expire=now --all
git gc --prune=now --aggressive

echo ""
echo "✅ 本地清理完成！"
echo ""
echo "⚠️  下一步：强制推送到 GitHub"
echo "   git push --force --all"
echo "   git push --force --tags"
echo ""
echo "📝 注意事项："
echo "   1. 通知所有协作者重新克隆仓库"
echo "   2. 在 GitHub Settings > Secrets 中更新所有密钥"
echo "   3. 撤销/更新所有泄露的凭证"

# 清理临时文件
rm /tmp/credentials-to-remove.txt
