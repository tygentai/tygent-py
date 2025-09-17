"use strict";
const vscode = require("vscode");
function buildInsertion(includeImport) {
    const importBlock = includeImport ? 'import tygent\n\n' : '';
    return `${importBlock}tygent.install()\n\n`;
}
function activate(context) {
    const disposable = vscode.commands.registerCommand('tygent.enableAgentCursor', async () => {
        const editor = vscode.window.activeTextEditor;
        if (!editor) {
            vscode.window.showWarningMessage('Open a Python file in Cursor to enable Tygent.');
            return;
        }
        const document = editor.document;
        if (document.languageId && document.languageId !== 'python') {
            vscode.window.showWarningMessage('The active file is not a Python file.');
            return;
        }
        const text = document.getText();
        if (text.includes('tygent.install()')) {
            vscode.window.showInformationMessage('Tygent is already installed in this file.');
            return;
        }
        const needsImport = !/^\s*import\s+tygent/m.test(text);
        let insertPosition = new vscode.Position(0, 0);
        let prefix = '';
        if (document.lineCount > 0) {
            const firstLine = document.lineAt(0);
            if (firstLine.text.startsWith('#!')) {
                insertPosition = firstLine.range.end;
                prefix = '\n';
            }
        }
        const insertion = prefix + buildInsertion(needsImport);
        const editApplied = await editor.edit((editBuilder) => {
            editBuilder.insert(insertPosition, insertion);
        });
        if (editApplied) {
            vscode.window.showInformationMessage('Tygent conversion snippet inserted for Cursor.');
        }
        else {
            vscode.window.showErrorMessage('Unable to modify document.');
        }
    });
    context.subscriptions.push(disposable);
}
exports.activate = activate;
function deactivate() { }
exports.deactivate = deactivate;
