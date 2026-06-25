// 通用 AJAX 表单提交
function submitForm(formId, url, successCallback) {
    $('#' + formId).submit(function(e) {
        e.preventDefault();
        var formData = new FormData(this);
        $.ajax({
            url: url,
            method: 'POST',
            data: formData,
            processData: false,
            contentType: false,
            success: successCallback,
            error: function(xhr) {
                alert('请求失败: ' + xhr.responseText);
            }
        });
    });
}

// 下载结果文件
function downloadResult(filename) {
    window.location.href = '/upload/download/' + filename;
}
