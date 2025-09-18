import * as vscode from 'vscode';

export function activate(context: vscode.ExtensionContext) {
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
        editor.edit(editBuilder => {
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

export function deactivate() {}
