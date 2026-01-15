# 🔒 安全事件处理清单

**事件时间**: 2026-01-15  
**事件类型**: Git 历史中包含敏感凭证  
**处理状态**: ✅ 已清理

---

## 📋 已完成的清理步骤

- [x] **步骤 1**: 使用 BFG 清理 Git 历史（32 个 commits）
- [x] **步骤 2**: 验证敏感数据已移除
- [x] **步骤 3**: 强制推送到 GitHub (`git push --force`)
- [x] **步骤 4**: GitHub 上的历史已更新

---

## ⚠️ 必须立即撤销的凭证

### 1. Email 应用密码
- **泄露内容**: `REDACTED_PASSWORD`
- **受影响账户**: `user@example.com`
- **操作步骤**:
  1. 前往 https://myaccount.google.com/apppasswords
  2. 找到名为 "stock-underdog-ml" 的应用密码
  3. 点击「删除」或「撤销」
  4. 重新生成新的应用密码
  5. 更新 `.env` 文件中的 `EMAIL_PASSWORD`

**状态**: [ ] 待处理

---

### 2. Telegram Bot Token
- **泄露内容**: `REDACTED_TOKEN`（部分）
- **Bot 名称**: （请填写）
- **操作步骤**:
  1. 打开 Telegram，找到 @BotFather
  2. 发送 `/mybots`
  3. 选择您的 bot
  4. 选择「API Token」
  5. 选择「Revoke current token」（撤销当前 token）
  6. 复制新的 token
  7. 更新 `.env` 文件中的 `TELEGRAM_BOT_TOKEN`

**状态**: [ ] 待处理

---

### 3. MongoDB Atlas 凭证
- **泄露内容**: `mongodb+srv://user:pass@host.mongodb.net/`
- **数据库**: cl.mongodb.net
- **用户**: david
- **操作步骤**:

**选项 A - 修改密码（推荐）**:
  1. 登入 https://cloud.mongodb.com
  2. 进入「Database Access」
  3. 找到用户 `david`
  4. 点击「Edit」
  5. 点击「Edit Password」
  6. 生成强密码或输入新密码
  7. 保存
  8. 更新 `.env` 文件中的 `MONGO_URI`

**选项 B - 删除用户并重建**:
  1. 删除用户 `david`
  2. 创建新用户（建议用不同名称）
  3. 设置权限
  4. 更新 `.env` 文件

**状态**: [ ] 待处理

---

### 4. Telegram Channel ID
- **泄露内容**: `-1001234567`（部分）
- **风险**: 低（只有 ID，无法直接操作）
- **建议**: 如果是私密频道，考虑创建新频道

**状态**: [ ] 可选

---

## 🛡️ 预防措施（已实施）

### Git 配置
- [x] `.gitignore` 已加强（包含凭证文件模式）
- [x] `example.env` 已改为占位符
- [x] `.env` 已被 gitignore
- [x] 添加了 `explain.md` 提醒 LLM 注意安全

### 代码审查
- [x] 检查所有 `.env*` 文件
- [x] 检查所有配置文件
- [x] 检查 `nohup.out` 等日志文件

---

## 📝 后续行动

### 立即（今天完成）
1. [ ] 撤销/更新所有泄露的凭证（见上方清单）
2. [ ] 测试新凭证是否正常工作
3. [ ] 在 GitHub 上关闭或回应 security alert

### 短期（本周完成）
1. [ ] 检查其他相关服务是否使用相同密码
2. [ ] 启用两步验证（如未启用）
3. [ ] 审查所有系统日志，确认无异常访问

### 长期（持续）
1. [ ] 定期轮换密码（建议 90 天）
2. [ ] 使用密码管理器（如 1Password, Bitwarden）
3. [ ] 考虑使用 GitHub Secrets 存储敏感数据
4. [ ] 设置 pre-commit hook 检测敏感数据

---

## 🔍 验证步骤

### 验证 Git 历史已清理
```bash
# 这些命令应该没有任何输出
git log --all -S "REDACTED_PASSWORD" --oneline
git log --all -S "REDACTED_TK" --oneline
git log --all -S "mongodb+srv://david:REDACTED" --oneline
```

### 验证 GitHub 状态
1. 前往 https://github.com/tbdavid2019/stock-underdog-ml/settings/security_analysis
2. 检查「Secret scanning」是否还有 alert
3. 如果还有，可能需要等待几分钟让 GitHub 重新扫描

---

## 📚 相关文档

- [GitHub Secret Scanning](https://docs.github.com/en/code-security/secret-scanning)
- [BFG Repo-Cleaner](https://rtyley.github.io/bfg-repo-cleaner/)
- [Git Filter-Repo](https://github.com/newren/git-filter-repo)

---

## ✅ 完成确认

请在完成所有步骤后填写：

- 处理人: _______________
- 完成日期: _______________
- 所有凭证已更新: [ ] 是 / [ ] 否
- 服务正常运行: [ ] 是 / [ ] 否
- GitHub alert 已关闭: [ ] 是 / [ ] 否

---

**最后更新**: 2026-01-15 12:38  
**处理工具**: BFG Repo-Cleaner 1.14.0
