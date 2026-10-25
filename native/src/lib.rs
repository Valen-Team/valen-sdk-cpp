//! Small C ABI over mature HF tokenizers and MiniJinja; no model math here.
use minijinja::{Environment, Error, ErrorKind};
use serde_json::{json, Value};
use std::ffi::{c_char, CStr, CString};
use tokenizers::Tokenizer;
struct Components {
    tokenizer: Tokenizer,
    environment: Environment<'static>,
}
fn output(value: Value) -> *mut c_char {
    CString::new(value.to_string()).unwrap().into_raw()
}
#[no_mangle]
pub unsafe extern "C" fn valen_components_new(
    tokenizer: *const c_char,
    template: *const c_char,
) -> *mut std::ffi::c_void {
    let create = || -> Result<Components, String> {
        let tokenizer = Tokenizer::from_bytes(CStr::from_ptr(tokenizer).to_bytes())
            .map_err(|e| e.to_string())?;
        let mut environment = Environment::new();
        environment
            .set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        environment.set_fuel(Some(1000000));
        environment.add_function(
            "raise_exception",
            |message: String| -> Result<String, Error> {
                Err(Error::new(ErrorKind::InvalidOperation, message))
            },
        );
        environment
            .add_template_owned(
                "chat",
                CStr::from_ptr(template)
                    .to_str()
                    .map_err(|e| e.to_string())?
                    .to_owned(),
            )
            .map_err(|e| e.to_string())?;
        Ok(Components {
            tokenizer,
            environment,
        })
    };
    match create() {
        Ok(c) => Box::into_raw(Box::new(c)) as _,
        Err(_) => std::ptr::null_mut(),
    }
}
#[no_mangle]
pub unsafe extern "C" fn valen_components_call(
    handle: *mut std::ffi::c_void,
    operation: *const c_char,
    input: *const c_char,
) -> *mut c_char {
    let c = &*(handle as *mut Components);
    let result = || -> Result<Value, String> {
        let op = CStr::from_ptr(operation)
            .to_str()
            .map_err(|e| e.to_string())?;
        let text = CStr::from_ptr(input).to_str().map_err(|e| e.to_string())?;
        match op {
            "encode" => {
                let e = c.tokenizer.encode(text, false).map_err(|e| e.to_string())?;
                Ok(json!({"ids":e.get_ids(),"offsets":e.get_offsets()}))
            }
            "render" => {
                let v: Value = serde_json::from_str(text).map_err(|e| e.to_string())?;
                Ok(
                    json!({"text":c.environment.get_template("chat").map_err(|e|e.to_string())?.render(minijinja::value::Serde(v)).map_err(|e|e.to_string())?}),
                )
            }
            _ => Err("Unknown component operation".into()),
        }
    };
    match result() {
        Ok(v) => output(v),
        Err(e) => output(json!({"error":e})),
    }
}
#[no_mangle]
pub unsafe extern "C" fn valen_components_free(handle: *mut std::ffi::c_void) {
    if !handle.is_null() {
        drop(Box::from_raw(handle as *mut Components));
    }
}
#[no_mangle]
pub unsafe extern "C" fn valen_string_free(s: *mut c_char) {
    if !s.is_null() {
        drop(CString::from_raw(s));
    }
}
