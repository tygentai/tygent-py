"use strict";
const vscode = require("vscode");
function activate(context) {
    const disposable = vscode.commands.registerCommand('tygent.enableAgent', () => {
        const editor = vscode.window.activeTextEditor;
        if (!editor) {
            return;
        }
        const doc = editor.document;
        const text = doc.getText();
        if (text.includes('tygent.install()')) {
            vscode.window.showInformationMessage('Tygent already enabled in this file.');
            return;
        }
        editor.edit((editBuilder) => {
            const firstLine = doc.lineAt(0);
            let position = new vscode.Position(0, 0);
            let prefix = '';
            if (firstLine.text.startsWith('#!')) {
                position = firstLine.range.end;
                prefix = '\n';
            }
            editBuilder.insert(position, `${prefix}import tygent\n\ntygent.install()\n\n`);
        });
    });
    context.subscriptions.push(disposable);
}
exports.activate = activate;
function deactivate() { }
exports.deactivate = deactivate;
