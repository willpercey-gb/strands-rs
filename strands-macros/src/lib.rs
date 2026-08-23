use proc_macro::TokenStream;

/// Attribute macro for defining agent tools from async functions.
///
/// The function's doc comment becomes the tool description. Parameter
/// descriptions come from a rustdoc `# Arguments` section — Rust rejects doc
/// comments in parameter position, so this is the only legal place to put them,
/// and it is where a Rust reader would look anyway.
///
/// # Example
///
/// ```ignore
/// /// Get the current weather for a city.
/// ///
/// /// # Arguments
/// ///
/// /// * `city` - The city to check weather for
/// /// * `unit` - Temperature unit (celsius or fahrenheit)
/// #[tool]
/// async fn get_weather(
///     city: String,
///     unit: Option<String>,
/// ) -> Result<String, strands_core::StrandsError> {
///     Ok(format!("22 degrees in {city}"))
/// }
/// ```
#[proc_macro_attribute]
pub fn tool(_attr: TokenStream, item: TokenStream) -> TokenStream {
    tool_impl(item.into()).into()
}

fn tool_impl(input: proc_macro2::TokenStream) -> proc_macro2::TokenStream {
    let func: syn::ItemFn = match syn::parse2(input.clone()) {
        Ok(f) => f,
        Err(e) => return e.to_compile_error(),
    };

    let fn_name = &func.sig.ident;
    let fn_name_str = fn_name.to_string();

    // Build PascalCase struct name
    let struct_name_str = to_pascal_case(&fn_name_str);
    let struct_name = syn::Ident::new(&struct_name_str, fn_name.span());

    // The full doc comment carries both the description and, by rustdoc
    // convention, the per-parameter documentation.
    let doc_lines = extract_doc_lines(&func.attrs);
    let description = doc_summary(&doc_lines);
    let arg_docs = parse_argument_docs(&doc_lines);

    // Parse parameters (skip self if present)
    let params: Vec<_> = func
        .sig
        .inputs
        .iter()
        .filter_map(|arg| {
            if let syn::FnArg::Typed(pat_type) = arg {
                let name = match pat_type.pat.as_ref() {
                    syn::Pat::Ident(ident) => ident.ident.to_string(),
                    _ => return None,
                };
                // Rust rejects doc comments on parameters, so they can only
                // come from the function's `# Arguments` section.
                let doc = arg_docs
                    .iter()
                    .find(|(param, _)| *param == name)
                    .map(|(_, text)| text.clone())
                    .unwrap_or_default();
                let ty = &pat_type.ty;
                let is_option = is_option_type(ty);
                let json_type = rust_type_to_json_type(ty);
                Some(ParamInfo {
                    name,
                    doc,
                    is_option,
                    json_type,
                    ty: ty.clone(),
                })
            } else {
                None
            }
        })
        .collect();

    // Build JSON schema properties
    let schema_properties: Vec<proc_macro2::TokenStream> = params
        .iter()
        .map(|p| {
            let name = &p.name;
            let json_type = &p.json_type;
            let desc = &p.doc;
            quote::quote! {
                properties.insert(
                    #name.to_string(),
                    serde_json::json!({
                        "type": #json_type,
                        "description": #desc
                    }),
                );
            }
        })
        .collect();

    let required_params: Vec<proc_macro2::TokenStream> = params
        .iter()
        .filter(|p| !p.is_option)
        .map(|p| {
            let name = &p.name;
            quote::quote! { #name.to_string() }
        })
        .collect();

    // Build parameter extraction in invoke()
    let param_extractions: Vec<proc_macro2::TokenStream> = params
        .iter()
        .map(|p| {
            let name_str = &p.name;
            let name_ident = syn::Ident::new(&p.name, proc_macro2::Span::call_site());
            let ty = &p.ty;
            if p.is_option {
                quote::quote! {
                    let #name_ident: #ty = input.get(#name_str)
                        .and_then(|v| serde_json::from_value(v.clone()).ok());
                }
            } else {
                quote::quote! {
                    let #name_ident: #ty = serde_json::from_value(
                        input.get(#name_str)
                            .cloned()
                            .ok_or_else(|| strands_core::StrandsError::Tool {
                                tool_name: #fn_name_str.to_string(),
                                message: format!("Missing required parameter: {}", #name_str),
                            })?
                    ).map_err(|e| strands_core::StrandsError::Tool {
                        tool_name: #fn_name_str.to_string(),
                        message: format!("Invalid parameter {}: {}", #name_str, e),
                    })?;
                }
            }
        })
        .collect();

    let param_names: Vec<syn::Ident> = params
        .iter()
        .map(|p| syn::Ident::new(&p.name, proc_macro2::Span::call_site()))
        .collect();

    let output = quote::quote! {
        // Keep the original function
        #func

        pub struct #struct_name;

        #[::strands_core::__macro_support::async_trait]
        impl strands_core::Tool for #struct_name {
            fn name(&self) -> &str {
                #fn_name_str
            }

            fn spec(&self) -> strands_core::types::tools::ToolSpec {
                let mut properties = serde_json::Map::new();
                #(#schema_properties)*

                // Built through the constructor rather than a struct literal:
                // a literal breaks every time ToolSpec gains a field, and
                // nothing in-tree exercises this macro to catch it.
                strands_core::types::tools::ToolSpec::new(
                    #fn_name_str,
                    #description,
                    serde_json::json!({
                        "type": "object",
                        "properties": serde_json::Value::Object(properties),
                        "required": vec![#(#required_params),*]
                    }),
                )
            }

            async fn invoke(
                &self,
                input: serde_json::Value,
                _ctx: &strands_core::ToolContext,
            ) -> strands_core::Result<strands_core::ToolOutput> {
                #(#param_extractions)*

                let result = #fn_name(#(#param_names),*).await?;
                let content = serde_json::to_value(result)
                    .map_err(|e| strands_core::StrandsError::Tool {
                        tool_name: #fn_name_str.to_string(),
                        message: e.to_string(),
                    })?;
                Ok(strands_core::ToolOutput {
                    content,
                    is_error: false,
                })
            }
        }
    };

    output
}

struct ParamInfo {
    name: String,
    doc: String,
    is_option: bool,
    json_type: String,
    ty: Box<syn::Type>,
}

fn to_pascal_case(s: &str) -> String {
    s.split('_')
        .map(|word| {
            let mut chars = word.chars();
            match chars.next() {
                Some(c) => c.to_uppercase().to_string() + &chars.collect::<String>(),
                None => String::new(),
            }
        })
        .collect()
}

/// The description: every line before the first `#` heading.
fn doc_summary(lines: &[String]) -> String {
    lines
        .iter()
        .take_while(|line| !line.trim_start().starts_with('#'))
        .map(String::as_str)
        .collect::<Vec<_>>()
        .join(" ")
        .trim()
        .to_string()
}

/// Parse a rustdoc `# Arguments` section into `(parameter, description)` pairs.
///
/// Accepts the conventional forms rustdoc uses:
/// `* \`name\` - text`, `- \`name\`: text`, and the same without backticks.
fn parse_argument_docs(lines: &[String]) -> Vec<(String, String)> {
    let mut pairs = Vec::new();
    let mut in_arguments = false;

    for line in lines {
        let trimmed = line.trim();

        if let Some(heading) = trimmed.strip_prefix('#') {
            // Any other heading ends the Arguments section.
            in_arguments = heading
                .trim_start_matches('#')
                .trim()
                .eq_ignore_ascii_case("arguments")
                || heading
                    .trim_start_matches('#')
                    .trim()
                    .eq_ignore_ascii_case("args");
            continue;
        }

        if !in_arguments {
            continue;
        }

        let Some(item) = trimmed
            .strip_prefix('*')
            .or_else(|| trimmed.strip_prefix('-'))
        else {
            continue;
        };
        let item = item.trim();

        // Split the name from its description on the first separator.
        let (name, text) = match item.find(" - ") {
            Some(index) => (&item[..index], &item[index + 3..]),
            None => match item.find(": ") {
                Some(index) => (&item[..index], &item[index + 2..]),
                None => continue,
            },
        };

        let name = name.trim().trim_matches('`').trim();
        if !name.is_empty() {
            pairs.push((name.to_string(), text.trim().to_string()));
        }
    }

    pairs
}

/// Collect a doc comment, preserving line structure.
///
/// Line breaks matter: the `# Arguments` section is parsed line by line, and
/// joining with spaces (as the description path does) makes it unreadable.
fn extract_doc_lines(attrs: &[syn::Attribute]) -> Vec<String> {
    attrs
        .iter()
        .filter_map(|attr| {
            if attr.path().is_ident("doc") {
                if let syn::Meta::NameValue(nv) = &attr.meta {
                    if let syn::Expr::Lit(expr_lit) = &nv.value {
                        if let syn::Lit::Str(s) = &expr_lit.lit {
                            return Some(s.value().trim().to_string());
                        }
                    }
                }
            }
            None
        })
        .collect()
}

fn is_option_type(ty: &syn::Type) -> bool {
    if let syn::Type::Path(type_path) = ty {
        if let Some(segment) = type_path.path.segments.last() {
            return segment.ident == "Option";
        }
    }
    false
}

fn rust_type_to_json_type(ty: &syn::Type) -> String {
    if let syn::Type::Path(type_path) = ty {
        if let Some(segment) = type_path.path.segments.last() {
            let ident = segment.ident.to_string();
            return match ident.as_str() {
                "String" | "str" => "string",
                "i8" | "i16" | "i32" | "i64" | "u8" | "u16" | "u32" | "u64" | "isize" | "usize" => {
                    "integer"
                }
                "f32" | "f64" => "number",
                "bool" => "boolean",
                "Vec" => "array",
                "Option" => {
                    // Unwrap the inner type
                    if let syn::PathArguments::AngleBracketed(args) = &segment.arguments {
                        if let Some(syn::GenericArgument::Type(inner)) = args.args.first() {
                            return rust_type_to_json_type(inner);
                        }
                    }
                    "string"
                }
                _ => "object",
            }
            .to_string();
        }
    }
    "string".to_string()
}
